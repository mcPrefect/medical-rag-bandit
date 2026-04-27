"""
Reward Sensitivity Analysis
Runs LinUCB bandit with four latency weights: 0.00, 0.05, 0.10, 0.20
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path
import sys
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.append(str(Path(__file__).parent.parent))

from retrieval.fast_arm import retrieve_fast
from retrieval.deep_arm import retrieve_deep
from bandit.linucb import LinUCB, extract_context
from llm.llm_wrapper import answer_question
from reward.reward_function import RewardFunction
from utils.config import load_config
from safety.validator import SafetyValidator

try:
    from retrieval.kg_arm import retrieve_kg, KnowledgeGraphArm
    KG_AVAILABLE = True
except Exception as e:
    print(f"[warn] KG arm not available: {e}. Graph arm will use fast fallback.")
    KG_AVAILABLE = False

# Weight redistribution

BASE_WEIGHTS = {
    "guideline": 0.55,
    "quality":   0.25,
    "latency":   0.10,
    "safety":    0.10,
}

LATENCY_CONDITIONS = [0.00, 0.05, 0.10, 0.20]


def redistribute_weights(target_latency: float) -> dict:
    """
    Set w_latency = target_latency and redistribute the delta proportionally
    across the other three components so that all weights still sum to 1.0.
    """
    delta = BASE_WEIGHTS["latency"] - target_latency          # positive if reducing latency weight
    others = {k: v for k, v in BASE_WEIGHTS.items() if k != "latency"}
    others_sum = sum(others.values())

    new_weights = {}
    for k, v in others.items():
        new_weights[k] = round(v + delta * (v / others_sum), 6)
    new_weights["latency"] = target_latency

    # Floating point safety: force exact sum to 1.0
    total = sum(new_weights.values())
    new_weights["guideline"] += round(1.0 - total, 8)

    return new_weights


def run_single_example(example, selected_arm, reward_fn, validator, config,
                        kg_arm=None):
    question  = example["QUESTION"]
    contexts  = example["CONTEXTS"]
    gold      = example["final_decision"]
    long_ans  = " ".join(example.get("LONG_ANSWER", example.get("long_answer", [])))

    # Retrieve
    t0 = time.time()
    top_k_fast  = config["retrieval"]["fast_arm"]["top_k"]
    top_k_deep  = config["retrieval"]["deep_arm"]["top_k"]
    top_k_kg    = config["retrieval"]["kg_arm"]["top_k"]

    if selected_arm == 0:
        retrieved = retrieve_fast(question, contexts, top_k=top_k_fast)
    elif selected_arm == 1:
        retrieved = retrieve_deep(question, contexts, top_k=top_k_deep, model_name=config['retrieval']['deep_arm']['model_name'])
    else:
        if KG_AVAILABLE and kg_arm is not None:
            retrieved = retrieve_kg(question, contexts, top_k=top_k_kg, kg_arm=kg_arm)
        else:
            # fallback: first-k sentences 
            sentences = []
            for ctx in contexts:
                sentences.extend(ctx.split(". "))
            retrieved = sentences[:top_k_kg]
    retrieval_time = time.time() - t0

    # LLM
    t0 = time.time()
    predicted, confidence = answer_question(question, retrieved,
                                max_new_tokens=config["llm"]["max_new_tokens"])
    llm_time = time.time() - t0

    # Safety
    is_safe, reason, _ = validator.validate(
        question=question,
        retrieved_context=retrieved,
        predicted_answer=predicted,
        confidence=confidence,
    )
    if not is_safe:
        predicted = "abstain"

    total_time = retrieval_time + llm_time
    correct = (predicted == gold)

    reward, components = reward_fn.compute_reward(
        predicted_answer=predicted,
        gold_answer=gold,
        generated_response=" ".join(retrieved),
        reference_text=long_ans,
        time_taken=total_time,
        safety_passed=is_safe,
    )

    return {
        "correct":        correct,
        "reward":         reward,
        "components":     components,
        "arm":            selected_arm,
        "total_time":     total_time,
        "retrieval_time": retrieval_time,
    }


# Condition runner
ARM_NAMES = {0: "fast", 1: "deep", 2: "graph"}


def run_condition(label: str, weights: dict, examples, config,
                  kg_arm=None) -> dict:
    """Run LinUCB bandit end-to-end for one latency weight condition."""
    print(f"  Condition: w_latency={weights['latency']:.2f}   ({label})")
    print(f"  Weights → guideline={weights['guideline']:.3f}  quality={weights['quality']:.3f}"
          f"  latency={weights['latency']:.3f}  safety={weights['safety']:.3f}")

    reward_fn = RewardFunction(
        w_guideline=weights["guideline"],
        w_quality=weights["quality"],
        w_latency=weights["latency"],
        w_safety=weights["safety"],
        time_budget=config["reward"]["time_budget"],
        safety_kill_switch=config["reward"]["safety_kill_switch"],
        use_bertscore=False,  # use sentence-transformers
    )

    validator = SafetyValidator(
        confidence_threshold=config["safety"]["confidence_threshold"],
        min_evidence_sentences=config["safety"]["min_evidence_sentences"],
    )

    bandit = LinUCB(
        n_arms=config["bandit"]["n_arms"],
        n_features=config["bandit"]["n_features"],
        alpha=config["bandit"]["alpha"],
    )

    arm_counts   = {0: 0, 1: 0, 2: 0}
    rewards      = []
    correct_list = []

    for i, example in enumerate(examples):
        question = example["QUESTION"]
        contexts = example["CONTEXTS"]

        ctx_vec  = extract_context(question, contexts, bandit=bandit, kg_arm=kg_arm)
        arm, _, _ = bandit.select_arm_with_probs(ctx_vec)
        arm = int(arm)
        arm_counts[int(arm)] += 1

        res = run_single_example(example, arm, reward_fn, validator, config, kg_arm)

        bandit.update(arm, ctx_vec, res["reward"])

        rewards.append(res["reward"])
        correct_list.append(int(res["correct"]))

        if (i + 1) % 100 == 0:
            acc = sum(correct_list) / len(correct_list)
            avg_r = np.mean(rewards)
            print(f"  [{i+1:>4}/{len(examples)}]  acc={acc:.1%}  avg_reward={avg_r:.4f}"
                  f"  fast={arm_counts[0]}  deep={arm_counts[1]}  graph={arm_counts[2]}")

    n = len(examples)
    result = {
        "label":       label,
        "weights":     weights,
        "n":           n,
        "accuracy":    sum(correct_list) / n,
        "avg_reward":  float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "arm_counts":  {ARM_NAMES[k]: v for k, v in arm_counts.items()},
        "arm_pct":     {ARM_NAMES[k]: round(v / n * 100, 1)
                        for k, v in arm_counts.items()},
    }

    print(f"\n  DONE - acc={result['accuracy']:.1%}  avg_reward={result['avg_reward']:.4f}")
    print(f"  Arm selections: fast={arm_counts[0]} ({result['arm_pct']['fast']}%)  "
          f"deep={arm_counts[1]} ({result['arm_pct']['deep']}%)  "
          f"graph={arm_counts[2]} ({result['arm_pct']['graph']}%)")

    return result



def print_table(results: list[dict]):
    print("Reward Sensitivity Analysis, Results Table")
    print(f"{'w_latency':>10}  {'Accuracy':>9}  {'Avg Reward':>11}  "
          f"{'Fast %':>8}  {'Deep %':>8}  {'Graph %':>9}")

    for r in results:
        wl  = r["weights"]["latency"]
        acc = r["accuracy"]
        avg = r["avg_reward"]
        fp  = r["arm_pct"]["fast"]
        dp  = r["arm_pct"]["deep"]
        gp  = r["arm_pct"]["graph"]

        print(f"{wl:>10.2f}  {acc:>8.1%}  {avg:>11.4f}  "
              f"{fp:>7.1f}%  {dp:>7.1f}%  {gp:>8.1f}%  ")


def main():
    parser = argparse.ArgumentParser(description="Reward sensitivity analysis")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of PubMedQA examples (default: 1000)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="results/reward_sensitivity.json")
    args = parser.parse_args()

    print("Reward Sensitivity Analysis")
    print(f"Latency weights to test: {LATENCY_CONDITIONS}")
    print(f"Examples per condition:  {args.n}")

    config = load_config(args.config)

    print("\nLoading PubMedQA data...")
    with open("data/pubmedqa/ori_pqal.json", "r") as f:
        data = json.load(f)
    examples = list(data.values())[: args.n]
    print(f"Loaded {len(examples)} examples.")

    kg_arm = None
    if KG_AVAILABLE:
        print("\nInitialising KG arm (shared across conditions)...")
        try:
            kg_arm = KnowledgeGraphArm(
                model_path=config.get("kg_arm", {}).get("model_path", "models/gnn_model_best.pt"),
                graph_path=config.get("kg_arm", {}).get("graph_path", "data/umls/subgraph.pkl"),
                concepts_path=config.get("kg_arm", {}).get("concepts_path", "data/umls/concepts.pkl"),
            )
            print("  KG arm ready.")
        except Exception as e:
            print(f"  KG arm init failed: {e} ,graph arm will use fast fallback.")

    # Run all conditions
    all_results = []
    for target_latency in LATENCY_CONDITIONS:
        weights = redistribute_weights(target_latency)
        label   = f"w_latency={target_latency:.2f}"
        result  = run_condition(label, weights, examples, config, kg_arm)
        all_results.append(result)

    print_table(all_results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "experiment": "reward_sensitivity",
                "n_examples": args.n,
                "latency_weights_tested": LATENCY_CONDITIONS,
                "base_weights": BASE_WEIGHTS,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()