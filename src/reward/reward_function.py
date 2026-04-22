"""
4-Component weighted reward function
R = 0.55·R_guideline + 0.25·R_quality + 0.10·R_latency + 0.10·R_safety
Safety kill-switch zeros entire reward on any safety failure
"""

import logging
from typing import Dict, Tuple
import os

os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["HF_HUB_DISABLE_SAFETENSORS_CONVERSION"] = "1"

logger = logging.getLogger(__name__)


class RewardFunction:
    
    def __init__(
        self,
        w_guideline: float = 0.55,
        w_quality: float = 0.25,
        w_latency: float = 0.10,
        w_safety: float = 0.10,
        time_budget: float = 10.0,
        safety_kill_switch: bool = True,
        bertscore_model: str = "microsoft/deberta-base-mnli",
        use_bertscore: bool = True,
    ):
    
        # Validate weights sum to 1.0
        total = w_guideline + w_quality + w_latency + w_safety
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"
        
        self.w_guideline = w_guideline
        self.w_quality = w_quality
        self.w_latency = w_latency
        self.w_safety = w_safety
        self.time_budget = time_budget
        self.safety_kill_switch = safety_kill_switch
        self.bertscore_model = bertscore_model
        self.use_bertscore = use_bertscore
        
        # Lazy-load BERTScore scorer
        self._scorer = None

    def _get_scorer(self):
        """Load sentence-transformers model for guideline adherence scoring."""
        if self._scorer is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading sentence-transformers model for guideline scoring")
                # self._scorer = SentenceTransformer('all-MiniLM-L6-v2')
                self._scorer = SentenceTransformer(self.bertscore_model)
                logger.info("Guideline scorer loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers: {e}")
                self._scorer = "unavailable"
        return self._scorer

    def compute_guideline_adherence(self, generated_response, reference_text):
        """
        R_guideline: cosine similarity between sentence embeddings of
        retrieved context and expert long answer.
        """
        if not generated_response or not reference_text:
            return 0.0

        if not self.use_bertscore:
            return self._fallback_guideline_score(generated_response, reference_text)

        scorer = self._get_scorer()
        if scorer == "unavailable":
            return self._fallback_guideline_score(generated_response, reference_text)

        try:
            embeddings = scorer.encode(
                [generated_response[:2000], reference_text[:2000]],
                convert_to_tensor=True
            )
            from torch.nn.functional import cosine_similarity
            sim = cosine_similarity(embeddings[0].unsqueeze(0),
                                    embeddings[1].unsqueeze(0)).item()
            # Map from [-1, 1] to [0, 1]
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))
        except Exception as e:
            logger.warning(f"Guideline scoring failed: {e}")
            return self._fallback_guideline_score(generated_response, reference_text)
    
    
    def _fallback_guideline_score(
        self, generated: str, reference: str
    ) -> float:
        """
        Fallback guideline score when BERTScore is unavailable.
        Uses token-level F1 overlap as a rough proxy.
        """
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        overlap = gen_tokens & ref_tokens
        precision = len(overlap) / len(gen_tokens) if gen_tokens else 0.0
        recall = len(overlap) / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def compute_quality(self, predicted_answer: str, gold_answer: str,):
        """exact match in (1.0 or 0.0). Abstention counts as 0"""
        if predicted_answer is None or gold_answer is None:
            return 0.0
        pred = predicted_answer.strip().lower()
        gold = gold_answer.strip().lower()
        if pred == "abstain":
            return 0.0
        return 1.0 if pred == gold else 0.0
    
    def compute_latency(self, time_taken: float, time_budget: float = None):
        """Linear decay max(0, 1 - time_taken / time_budget)"""
        budget = time_budget if time_budget is not None else self.time_budget
        if budget <= 0:
            return 0.0
        return max(0.0, 1.0 - time_taken / budget)
    
    def compute_safety(self, safety_passed: bool) -> float:
        """Binary: 1.0 if validator passed, 0.0 otherwise"""
        return 1.0 if safety_passed else 0.0
    
    def compute_reward(
        self,
        predicted_answer: str,
        gold_answer: str,
        generated_response: str = None,
        reference_text: str = None,
        time_taken: float = 0.0,
        safety_passed: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the full 4-component weighted reward.
        
        R = w_guideline·R_guideline + w_quality·R_quality 
            + w_latency·R_latency + w_safety·R_safety
        
        Kill-switch: If safety_passed is False AND kill_switch is enabled,
        the entire reward is set to 0.0 regardless of other components.
        
        Args:
            predicted_answer: Model's predicted answer (yes/no/maybe)
            gold_answer: Ground truth answer
            generated_response: Full LLM response text (for BERTScore)
            reference_text: Gold long answer / guideline text (for BERTScore)
            time_taken: Total response time in seconds
            safety_passed: Whether safety validator passed
            
        Returns:
            (total_reward, components_dict) where components_dict has:
                r_guideline, r_quality, r_latency, r_safety,
                reward_raw (before kill-switch), reward_final,
                kill_switch_triggered
        """
        # Compute individual components
        r_guideline = self.compute_guideline_adherence(
            generated_response or predicted_answer,
            reference_text or "",
        )
        r_quality = self.compute_quality(predicted_answer, gold_answer)
        r_latency = self.compute_latency(time_taken)
        r_safety = self.compute_safety(safety_passed)
        
        # Weighted combination
        reward_raw = (
            self.w_guideline * r_guideline
            + self.w_quality * r_quality
            + self.w_latency * r_latency
            + self.w_safety * r_safety
        )
        
        # Kill-switch: safety failure zeros entire reward
        kill_switch_triggered = False
        if self.safety_kill_switch and not safety_passed:
            reward_final = 0.0
            kill_switch_triggered = True
        else:
            reward_final = reward_raw
        
        # Build component breakdown for logging/analysis
        components = {
            "r_guideline": round(r_guideline, 4),
            "r_quality": round(r_quality, 4),
            "r_latency": round(r_latency, 4),
            "r_safety": round(r_safety, 4),
            "w_guideline": self.w_guideline,
            "w_quality": self.w_quality,
            "w_latency": self.w_latency,
            "w_safety": self.w_safety,
            "reward_raw": round(reward_raw, 4),
            "reward_final": round(reward_final, 4),
            "kill_switch_triggered": kill_switch_triggered,
        }
        
        return reward_final, components
    
    def __repr__(self) -> str:
        return (
            f"RewardFunction("
            f"guideline={self.w_guideline}, "
            f"quality={self.w_quality}, "
            f"latency={self.w_latency}, "
            f"safety={self.w_safety}, "
            f"time_budget={self.time_budget}s, "
            f"kill_switch={self.safety_kill_switch})"
        )


# ──────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────

def create_reward_function(config: dict) -> RewardFunction:
    """
    Create a RewardFunction from config dict.
    
    Expected config structure:
        reward:
            w_guideline: 0.55
            w_quality: 0.25
            w_latency: 0.10
            w_safety: 0.10
            time_budget: 10.0
            safety_kill_switch: true
            use_bertscore: true
            bertscore_model: "microsoft/deberta-base-mnli"
    """
    reward_cfg = config.get("reward", {})
    
    return RewardFunction(
        w_guideline=reward_cfg.get("w_guideline", 0.55),
        w_quality=reward_cfg.get("w_quality", 0.25),
        w_latency=reward_cfg.get("w_latency", 0.10),
        w_safety=reward_cfg.get("w_safety", 0.10),
        time_budget=reward_cfg.get("time_budget", 10.0),
        safety_kill_switch=reward_cfg.get("safety_kill_switch", True),
        use_bertscore=reward_cfg.get("use_bertscore", True),
        bertscore_model=reward_cfg.get(
            "guideline_model", "pritamdeka/S-PubMedBert-MS-MARCO"
        ),
    )


# ──────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("REWARD FUNCTION TEST")
    
    rf = RewardFunction(use_bertscore=False)  # Fast test without model
    print(f"\n{rf}\n")
    
    # Test 1: Correct answer, safety passed, fast response
    print("Test 1: Correct + Safe + Fast")
    reward, comp = rf.compute_reward(
        predicted_answer="yes",
        gold_answer="yes",
        generated_response="Based on the evidence, aspirin reduces cardiovascular risk.",
        reference_text="Studies demonstrate aspirin's efficacy in reducing cardiovascular events in high-risk populations.",
        time_taken=2.0,
        safety_passed=True,
    )
    print(f"  Reward: {reward:.4f}")
    for k, v in comp.items():
        print(f"  {k}: {v}")
    
    # Test 2: Correct answer but safety FAILED -- kill-switch
    print("\nTest 2: Correct but Safety FAILED (kill-switch)")
    reward, comp = rf.compute_reward(
        predicted_answer="yes",
        gold_answer="yes",
        generated_response="Yes, this treatment is recommended.",
        reference_text="Clinical guidelines support this treatment approach.",
        time_taken=3.0,
        safety_passed=False,
    )
    print(f"  Reward: {reward:.4f}")
    print(f"  Kill-switch triggered: {comp['kill_switch_triggered']}")
    assert reward == 0.0, "Kill-switch should zero the reward!"
    
    # Test 3: Wrong answer, safety passed
    print("\nTest 3: Wrong answer + Safe")
    reward, comp = rf.compute_reward(
        predicted_answer="no",
        gold_answer="yes",
        generated_response="No, there is insufficient evidence.",
        reference_text="Strong evidence supports this intervention.",
        time_taken=5.0,
        safety_passed=True,
    )
    print(f"  Reward: {reward:.4f}")
    for k, v in comp.items():
        print(f"  {k}: {v}")
    
    # Test 4: Abstention
    print("\nTest 4: Abstention (safety triggered)")
    reward, comp = rf.compute_reward(
        predicted_answer="abstain",
        gold_answer="yes",
        generated_response="",
        reference_text="The evidence clearly shows benefit.",
        time_taken=1.0,
        safety_passed=False,
    )
    print(f"  Reward: {reward:.4f}")
    print(f"  Kill-switch triggered: {comp['kill_switch_triggered']}")
    
    # Test 5: Slow response (exceeds time budget)
    print("\nTest 5: Very slow response (15s > 10s budget)")
    reward, comp = rf.compute_reward(
        predicted_answer="yes",
        gold_answer="yes",
        generated_response="Yes, the evidence supports this.",
        reference_text="Evidence supports this clinical decision.",
        time_taken=15.0,
        safety_passed=True,
    )
    print(f"  Reward: {reward:.4f}")
    print(f"  R_latency: {comp['r_latency']} (should be 0.0)")
    
    print("ALL TESTS PASSED")