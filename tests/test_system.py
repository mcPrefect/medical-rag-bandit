"""
System Test Suite: Verifies all components of the Autonomous Medical RAG Pipeline

Run with:
    pytest tests/test_system.py -v
    pytest tests/test_system.py -v -k "test_reward"     # just reward tests
    pytest tests/test_system.py -v -k "test_bandit"      # just bandit tests

Covers:
    1. Reward Function — 4 components, weights, kill-switch
    2. Context Features — 10 dimensions, value ranges
    3. Adaptive Alpha — decay formula, convergence
    4. Off-Policy Learning — IPS, bootstrap CI, policy comparison
    5. Safety Validator — abstention, contraindications
    6. Integration — components work together end-to-end
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reward.reward_function import RewardFunction, create_reward_function
from src.bandit.linucb import LinUCB, extract_context
from src.safety.validator import SafetyValidator
from src.learning.off_policy import (
    compute_ips,
    bootstrap_ci,
    compare_policies,
    make_always_arm_policy,
    make_uniform_policy,
    make_linucb_policy,
)



# 1. REWARD FUNCTION TESTS 

class TestRewardFunction:
    """Tests for the 4-component weighted reward (Section 3.3.2)."""

    def setup_method(self):
        """Create reward function with BERTScore disabled for fast tests."""
        self.rf = RewardFunction(use_bertscore=False)

    def test_weights_sum_to_one(self):
        """Weights must sum to 1.0 — otherwise reward scale is wrong."""
        total = self.rf.w_guideline + self.rf.w_quality + self.rf.w_latency + self.rf.w_safety
        assert abs(total - 1.0) < 1e-6

    def test_weights_match_report(self):
        """Weights must match interim report Section 3.3.2."""
        assert self.rf.w_guideline == 0.55
        assert self.rf.w_quality == 0.25
        assert self.rf.w_latency == 0.10
        assert self.rf.w_safety == 0.10

    def test_invalid_weights_rejected(self):
        """Weights that don't sum to 1.0 should raise an error."""
        with pytest.raises(AssertionError):
            RewardFunction(w_guideline=0.5, w_quality=0.5, w_latency=0.5, w_safety=0.5)

    def test_correct_answer_positive_reward(self):
        """Correct + safe + fast should produce positive reward."""
        reward, comp = self.rf.compute_reward(
            predicted_answer="yes", gold_answer="yes",
            time_taken=2.0, safety_passed=True,
        )
        assert reward > 0

    def test_wrong_answer_lower_reward(self):
        """Wrong answer should produce lower reward than correct."""
        r_correct, _ = self.rf.compute_reward(
            predicted_answer="yes", gold_answer="yes",
            time_taken=2.0, safety_passed=True,
        )
        r_wrong, _ = self.rf.compute_reward(
            predicted_answer="no", gold_answer="yes",
            time_taken=2.0, safety_passed=True,
        )
        assert r_correct > r_wrong

    def test_kill_switch_zeros_reward(self):
        """Safety failure must zero the entire reward (Section 3.3.2)."""
        reward, comp = self.rf.compute_reward(
            predicted_answer="yes", gold_answer="yes",
            time_taken=1.0, safety_passed=False,
        )
        assert reward == 0.0
        assert comp['kill_switch_triggered'] is True

    def test_kill_switch_disabled(self):
        """With kill-switch off, safety failure shouldn't zero reward."""
        rf_no_kill = RewardFunction(safety_kill_switch=False, use_bertscore=False)
        reward, comp = rf_no_kill.compute_reward(
            predicted_answer="yes", gold_answer="yes",
            time_taken=1.0, safety_passed=False,
        )
        assert reward > 0
        assert comp['kill_switch_triggered'] is False

    def test_quality_exact_match(self):
        """R_quality should be 1.0 for exact match, 0.0 otherwise."""
        assert self.rf.compute_quality("yes", "yes") == 1.0
        assert self.rf.compute_quality("no", "yes") == 0.0
        assert self.rf.compute_quality("maybe", "maybe") == 1.0

    def test_quality_abstention_is_zero(self):
        """Abstention should always score 0 on quality."""
        assert self.rf.compute_quality("abstain", "yes") == 0.0

    def test_latency_within_budget(self):
        """Fast response should score high on latency."""
        score = self.rf.compute_latency(1.0, time_budget=10.0)
        assert score == 0.9

    def test_latency_at_budget(self):
        """Response at exactly the budget should score 0."""
        score = self.rf.compute_latency(10.0, time_budget=10.0)
        assert score == 0.0

    def test_latency_over_budget(self):
        """Response over budget should score 0, not negative."""
        score = self.rf.compute_latency(15.0, time_budget=10.0)
        assert score == 0.0

    def test_safety_binary(self):
        """R_safety is strictly 1.0 or 0.0."""
        assert self.rf.compute_safety(True) == 1.0
        assert self.rf.compute_safety(False) == 0.0

    def test_reward_components_all_present(self):
        """compute_reward must return all expected component keys."""
        _, comp = self.rf.compute_reward(
            predicted_answer="yes", gold_answer="yes",
            time_taken=1.0, safety_passed=True,
        )
        expected_keys = {
            'r_guideline', 'r_quality', 'r_latency', 'r_safety',
            'w_guideline', 'w_quality', 'w_latency', 'w_safety',
            'reward_raw', 'reward_final', 'kill_switch_triggered',
        }
        assert set(comp.keys()) == expected_keys

    def test_reward_bounded(self):
        """Reward should be in [0, 1] range."""
        reward, _ = self.rf.compute_reward(
            predicted_answer="yes", gold_answer="yes",
            generated_response="yes", reference_text="yes this is correct",
            time_taken=0.1, safety_passed=True,
        )
        assert 0.0 <= reward <= 1.0

    def test_create_from_config(self):
        """Factory function should create reward from config dict."""
        config = {
            'reward': {
                'w_guideline': 0.55, 'w_quality': 0.25,
                'w_latency': 0.10, 'w_safety': 0.10,
                'time_budget': 10.0, 'safety_kill_switch': True,
                'use_bertscore': False,
            }
        }
        rf = create_reward_function(config)
        assert rf.w_guideline == 0.55
        assert rf.time_budget == 10.0


# 2. CONTEXT FEATURES TESTS (Task 2)

class TestContextFeatures:
    """Tests for the 10-dimensional context vector (Section 3.3.1)."""

    def test_output_shape(self):
        """Must return exactly 10 features."""
        features = extract_context("Does aspirin help?", ["Context sentence one.", "Context sentence two."])
        assert features.shape == (10,)

    def test_all_features_normalised(self):
        """All features should be in [0, 1] range."""
        features = extract_context(
            "Is emergency treatment needed for elderly pregnant patient with renal failure?",
            ["Patient presents with acute symptoms.", "History of kidney disease."],
        )
        for i, val in enumerate(features):
            assert 0.0 <= val <= 1.0, f"Feature {i+1} out of range: {val}"

    def test_urgency_detected(self):
        """Emergency keywords should produce non-zero urgency (feature 2)."""
        features = extract_context(
            "Is emergency chest pain treatment needed?",
            ["Patient has acute symptoms."],
        )
        assert features[1] > 0, "Urgency should be > 0 for emergency query"

    def test_no_urgency_for_routine(self):
        """Routine query should have zero urgency."""
        features = extract_context(
            "Does vitamin D supplementation help?",
            ["Studies show moderate benefit."],
        )
        assert features[1] == 0.0

    def test_patient_risk_detected(self):
        """High-risk descriptors should produce non-zero risk (feature 3)."""
        features = extract_context(
            "Treatment for elderly immunocompromised patient?",
            ["Patient is on dialysis."],
        )
        assert features[2] > 0, "Patient risk should be > 0"

    def test_question_length_scales(self):
        """Longer questions should produce higher feature 4."""
        short = extract_context("Is aspirin safe?", ["Context."])
        long = extract_context(
            "Is aspirin safe for long-term use in elderly patients with multiple comorbidities and renal impairment?",
            ["Context."],
        )
        assert long[3] > short[3]

    def test_guideline_coverage_detected(self):
        """Queries mentioning guideline topics should score on feature 8."""
        features = extract_context(
            "Is aspirin contraindicated with warfarin for patients with asthma?",
            ["Guidelines recommend caution with anticoagulant use."],
        )
        assert features[7] > 0, "Guideline coverage should be > 0"

    def test_historical_performance_default(self):
        """Without bandit, historical arm performance should default to 0.5."""
        features = extract_context("Test question?", ["Context."])
        assert features[8] == 0.5

    def test_historical_performance_with_bandit(self):
        """With bandit, feature 9 should reflect arm performance."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        # Give arm 0 some good rewards
        ctx = np.random.random(10)
        for _ in range(10):
            bandit.update(0, ctx, 0.9)
        features = extract_context("Test question?", ["Context."], bandit=bandit)
        assert features[8] > 0.5, "Should reflect good arm performance"

    def test_kg_density_default_zero(self):
        """Without KG arm, feature 10 should be 0."""
        features = extract_context("Test question?", ["Context."])
        assert features[9] == 0.0

    def test_empty_context(self):
        """Should handle empty context list without crashing."""
        features = extract_context("Test question?", [])
        assert features.shape == (10,)
        assert features[4] == 0.0  # n_contexts should be 0


# 3. ADAPTIVE ALPHA TESTS (Task 3)

class TestAdaptiveAlpha:
    """Tests for alpha decay: α_t = α_0 / √t (Section 3.3.1)."""

    def test_initial_alpha(self):
        """Alpha should start at α_0."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        assert bandit.alpha == 2.0

    def test_alpha_decays(self):
        """Alpha should decrease after multiple updates."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        initial = bandit.alpha
        ctx = np.random.random(10)
        bandit.update(0, ctx, 1.0)
        bandit.update(0, ctx, 1.0)  # After 2 steps: α = 2.0/√2 ≈ 1.414
        assert bandit.alpha < initial

    def test_alpha_formula(self):
        """α_t = α_0 / √t should hold exactly."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        for _ in range(100):
            bandit.update(0, ctx, 0.5)
        expected = 2.0 / np.sqrt(100)
        assert abs(bandit.alpha - expected) < 1e-6

    def test_alpha_convergence(self):
        """After many steps, alpha should be small but positive."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        for _ in range(10000):
            bandit.update(0, ctx, 0.5)
        assert 0 < bandit.alpha < 0.1

    def test_alpha_never_zero(self):
        """Alpha should never reach exactly zero."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        for _ in range(100000):
            bandit.update(0, ctx, 0.5)
        assert bandit.alpha > 0


# 4. LINUCB BANDIT TESTS

class TestLinUCB:
    """Tests for LinUCB bandit core functionality."""

    def test_select_arm_returns_valid(self):
        """Selected arm must be in range [0, n_arms)."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        arm = bandit.select_arm(ctx)
        assert 0 <= arm < 3

    def test_select_arm_with_probs_sums_to_one(self):
        """Arm probabilities must sum to 1.0."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        arm, probs, scores = bandit.select_arm_with_probs(ctx)
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_select_arm_with_probs_all_positive(self):
        """All arm probabilities must be > 0 (softmax guarantees this)."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        _, probs, _ = bandit.select_arm_with_probs(ctx)
        assert all(p > 0 for p in probs)

    def test_selected_arm_has_highest_ucb(self):
        """Selected arm should correspond to highest UCB score."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        arm, probs, scores = bandit.select_arm_with_probs(ctx)
        assert arm == np.argmax(scores)

    def test_get_action_probabilities(self):
        """get_action_probabilities should match select_arm_with_probs."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        _, probs1, _ = bandit.select_arm_with_probs(ctx)
        probs2 = bandit.get_action_probabilities(ctx)
        np.testing.assert_array_almost_equal(probs1, probs2)

    def test_dimension_mismatch_handled(self):
        """Bandit should handle context of wrong dimension without crashing."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        # Too short — should pad
        arm = bandit.select_arm(np.random.random(4))
        assert 0 <= arm < 3
        # Too long — should truncate
        arm = bandit.select_arm(np.random.random(15))
        assert 0 <= arm < 3

    def test_save_load_weights(self, tmp_path):
        """Save and load should preserve bandit state."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        ctx = np.random.random(10)
        # Train for a bit
        for _ in range(50):
            arm = bandit.select_arm(ctx)
            bandit.update(arm, ctx, np.random.random())

        # Save
        path = str(tmp_path / "weights.pkl")
        bandit.save_weights(path)

        # Load into fresh bandit
        bandit2 = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        assert bandit2.load_weights(path) is True

        # Verify state matches
        assert bandit2.t == bandit.t
        assert bandit2.alpha_0 == bandit.alpha_0
        for i in range(3):
            np.testing.assert_array_almost_equal(bandit2.A[i], bandit.A[i])
            np.testing.assert_array_almost_equal(bandit2.b[i], bandit.b[i])

    def test_load_nonexistent_returns_false(self, tmp_path):
        """Loading from missing file should return False, not crash."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        assert bandit.load_weights(str(tmp_path / "nope.pkl")) is False

    def test_step_counter_increments(self):
        """Step counter should increment on each update."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        assert bandit.t == 0
        bandit.update(0, np.random.random(10), 0.5)
        assert bandit.t == 1
        bandit.update(1, np.random.random(10), 0.3)
        assert bandit.t == 2


# 5. OFF-POLICY LEARNING TESTS (Task 4)

class TestOffPolicyLearning:
    """Tests for IPS estimator and policy comparison (Section 3.6)."""

    def _make_synthetic_log(self, n=100):
        """Create synthetic log data for testing."""
        np.random.seed(42)
        log = []
        for _ in range(n):
            log.append({
                'context_vector': np.random.random(10).tolist(),
                'selected_arm': np.random.choice(3),
                'reward': np.random.random(),
                'arm_probabilities': [0.5, 0.3, 0.2],
            })
        return log

    def test_ips_returns_float(self):
        """IPS estimate should be a single float."""
        log = self._make_synthetic_log()
        policy = make_uniform_policy(3)
        v_ips, wr = compute_ips(log, policy)
        assert isinstance(v_ips, float)

    def test_ips_weighted_rewards_length(self):
        """Should return one weighted reward per log entry."""
        log = self._make_synthetic_log(50)
        policy = make_uniform_policy(3)
        _, wr = compute_ips(log, policy)
        assert len(wr) == 50

    def test_ips_same_policy_recovers_mean(self):
        """IPS with logging policy should approximate observed mean reward."""
        log = self._make_synthetic_log(500)
        # Use the same probabilities as the logging policy
        def same_policy(ctx):
            return np.array([0.5, 0.3, 0.2])
        v_ips, _ = compute_ips(log, same_policy, cap=100)  # High cap to avoid bias
        actual_mean = np.mean([e['reward'] for e in log])
        # Should be close (not exact due to discrete arm selection)
        assert abs(v_ips - actual_mean) < 0.15

    def test_ips_capping_reduces_extreme_weights(self):
        """Capped IPS should have lower variance than uncapped."""
        log = self._make_synthetic_log(200)
        # Policy that strongly disagrees with logging policy
        extreme_policy = make_always_arm_policy(2, 3)  # Always arm 2, but logged at 0.2 prob
        _, wr_capped = compute_ips(log, extreme_policy, cap=5.0)
        _, wr_uncapped = compute_ips(log, extreme_policy, cap=1000.0)
        assert np.std(wr_capped) <= np.std(wr_uncapped)

    def test_ips_empty_log(self):
        """Empty log should return 0 without crashing."""
        policy = make_uniform_policy(3)
        v_ips, wr = compute_ips([], policy)
        assert v_ips == 0.0
        assert len(wr) == 0

    def test_bootstrap_ci_contains_mean(self):
        """95% CI should contain the mean estimate."""
        log = self._make_synthetic_log(100)
        policy = make_uniform_policy(3)
        _, wr = compute_ips(log, policy)
        lower, mean, upper = bootstrap_ci(wr)
        assert lower <= mean <= upper

    def test_bootstrap_ci_width(self):
        """CI should have positive width."""
        log = self._make_synthetic_log(100)
        policy = make_uniform_policy(3)
        _, wr = compute_ips(log, policy)
        lower, _, upper = bootstrap_ci(wr)
        assert upper > lower

    def test_always_arm_policy_deterministic(self):
        """Always-arm policy should assign 1.0 to chosen arm."""
        policy = make_always_arm_policy(1, 3)
        probs = policy(np.random.random(10))
        assert probs[1] == 1.0
        assert probs[0] == 0.0
        assert probs[2] == 0.0

    def test_uniform_policy_equal(self):
        """Uniform policy should assign equal probability."""
        policy = make_uniform_policy(3)
        probs = policy(np.random.random(10))
        np.testing.assert_array_almost_equal(probs, [1/3, 1/3, 1/3])

    def test_compare_policies_structure(self):
        """Policy comparison should return all expected keys."""
        log = self._make_synthetic_log(100)
        result = compare_policies(
            log, make_uniform_policy(3), make_always_arm_policy(0, 3),
        )
        expected_keys = {
            'v_current', 'v_candidate', 'improvement', 'improvement_pct',
            'ci_current', 'ci_candidate', 'ci_improvement',
            'meets_threshold', 'ci_excludes_negative', 'recommend_update',
        }
        assert set(result.keys()) == expected_keys

    def test_compare_same_policy_no_improvement(self):
        """Comparing a policy to itself should show ~0 improvement."""
        log = self._make_synthetic_log(100)
        policy = make_uniform_policy(3)
        result = compare_policies(log, policy, policy)
        assert abs(result['improvement']) < 0.01

    def test_linucb_policy_wrapper(self):
        """LinUCB policy wrapper should return valid probabilities."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        policy = make_linucb_policy(bandit)
        probs = policy(np.random.random(10))
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(p > 0 for p in probs)


# 6. SAFETY VALIDATOR TESTS

class TestSafetyValidator:
    """Tests for the multi-layer safety validator."""

    def setup_method(self):
        self.validator = SafetyValidator(
            confidence_threshold=0.7,
            min_evidence_sentences=2,
        )

    def test_passes_valid_input(self):
        """Normal valid input should pass all checks."""
        safe, reason, _ = self.validator.validate(
            question="Does aspirin help?",
            retrieved_context=[
                "Studies show aspirin reduces heart attack risk significantly.",
                "Daily low-dose aspirin is recommended for cardiovascular prevention.",
            ],
            predicted_answer="yes",
            confidence=0.9,
        )
        assert safe is True

    def test_rejects_low_confidence(self):
        """Low confidence should trigger abstention."""
        safe, reason, _ = self.validator.validate(
            question="Does this work?",
            retrieved_context=["Evidence one.", "Evidence two."],
            predicted_answer="yes",
            confidence=0.3,
        )
        assert safe is False
        assert "confidence" in reason.lower()

    def test_rejects_insufficient_evidence(self):
        """Too few context sentences should trigger abstention."""
        safe, reason, _ = self.validator.validate(
            question="Does this work?",
            retrieved_context=["Only one sentence."],
            predicted_answer="yes",
            confidence=0.9,
        )
        assert safe is False
        assert "evidence" in reason.lower()

    def test_detects_contraindication(self):
        """Known drug-condition pair should be flagged."""
        safe, reason, _ = self.validator.validate(
            question="Patient has bleeding disorder, should we use aspirin?",
            retrieved_context=[
                "Aspirin is a common analgesic.",
                "Patient has history of bleeding.",
            ],
            predicted_answer="yes",
            confidence=0.9,
        )
        assert safe is False
        assert "contraindication" in reason.lower()

    def test_rejects_invalid_answer_format(self):
        """Answers other than yes/no/maybe should fail sanity check."""
        safe, reason, _ = self.validator.validate(
            question="Is this effective?",
            retrieved_context=["Study one shows benefit.", "Study two confirms."],
            predicted_answer="absolutely",
            confidence=0.9,
        )
        assert safe is False

    def test_no_confidence_defaults_pass(self):
        """No confidence score should default to passing."""
        safe, _, details = self.validator.validate(
            question="Does this help?",
            retrieved_context=["Evidence one.", "Evidence two."],
            predicted_answer="yes",
            confidence=None,
        )
        assert details['confidence']['pass'] is True


# 7. INTEGRATION TESTS

class TestIntegration:
    """Tests that components work together correctly."""

    def test_reward_with_safety_kill_switch(self):
        """Safety validator failure → kill-switch → zero reward."""
        validator = SafetyValidator(confidence_threshold=0.7)
        rf = RewardFunction(use_bertscore=False)

        # Validator should reject low confidence
        safe, _, _ = validator.validate(
            question="Is this safe?",
            retrieved_context=["Evidence one.", "Evidence two."],
            predicted_answer="yes",
            confidence=0.3,
        )
        assert safe is False

        # Reward should be zeroed
        reward, comp = rf.compute_reward(
            predicted_answer="yes", gold_answer="yes",
            time_taken=1.0, safety_passed=safe,
        )
        assert reward == 0.0
        assert comp['kill_switch_triggered'] is True

    def test_bandit_features_to_selection_to_reward(self):
        """Full flow: extract features → select arm → compute reward."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        rf = RewardFunction(use_bertscore=False)

        question = "Does aspirin reduce cardiovascular risk?"
        contexts = ["Studies show benefit.", "Large trials confirm efficacy."]

        # Extract features
        features = extract_context(question, contexts, bandit=bandit)
        assert features.shape == (10,)

        # Select arm
        arm, probs, scores = bandit.select_arm_with_probs(features)
        assert 0 <= arm < 3
        assert abs(sum(probs) - 1.0) < 1e-6

        # Compute reward
        reward, comp = rf.compute_reward(
            predicted_answer="yes", gold_answer="yes",
            time_taken=2.0, safety_passed=True,
        )
        assert reward > 0

        # Update bandit
        bandit.update(arm, features, reward)
        assert bandit.t == 1

    def test_offpolicy_on_bandit_decisions(self):
        """IPS should work on data generated by a real bandit."""
        bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
        rf = RewardFunction(use_bertscore=False)

        # Simulate a few decisions
        log = []
        for _ in range(20):
            ctx = np.random.random(10)
            arm, probs, _ = bandit.select_arm_with_probs(ctx)
            reward = np.random.random()
            bandit.update(arm, ctx, reward)

            log.append({
                'context_vector': ctx.tolist(),
                'selected_arm': arm,
                'reward': reward,
                'arm_probabilities': probs.tolist(),
            })

        # Run IPS — should not crash
        policy = make_uniform_policy(3)
        v_ips, wr = compute_ips(log, policy)
        assert isinstance(v_ips, float)
        assert len(wr) == 20

# Run with: pytest tests/test_system.py -v

