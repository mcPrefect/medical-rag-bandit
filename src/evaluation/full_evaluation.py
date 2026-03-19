"""
Comprehensive Evaluation: Full 1000-example evaluation with baselines,
statistical tests, ablation studies, and error analysis.

Covers Task 6A-F from the implementation plan.

Usage:
    python src/evaluation/full_evaluation.py
    python src/evaluation/full_evaluation.py --n 200  # quick test
"""

import transformers.safetensors_conversion
transformers.safetensors_conversion.auto_conversion = lambda *args, **kwargs: None

import threading
_original_start = threading.Thread.start
def _patched_start(self):
    if 'auto_conversion' in str(self._target):
        return    
    _original_start(self)
threading.Thread.start = _patched_start

import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from retrieval.fast_arm import retrieve_fast
from retrieval.deep_arm import retrieve_deep
from retrieval.kg_arm import KnowledgeGraphArm, retrieve_kg
from bandit.linucb import LinUCB, extract_context
from llm.llm_wrapper import answer_question
from safety.validator import SafetyValidator
from reward.reward_function import RewardFunction, create_reward_function
from utils.config import load_config


# shared state so we only load heavy models once
KG_ARM = None
REWARD_FN = None
VALIDATOR = None
CONFIG = None


def init_shared(config_path="configs/config.yaml"):
    global KG_ARM, REWARD_FN, VALIDATOR, CONFIG
    CONFIG = load_config(config_path)
    print("Loading KG arm...")
    KG_ARM = KnowledgeGraphArm()
    REWARD_FN = create_reward_function(CONFIG)
    REWARD_FN.use_bertscore=False
    VALIDATOR = SafetyValidator(
        confidence_threshold=CONFIG['safety']['confidence_threshold'],
        min_evidence_sentences=CONFIG['safety']['min_evidence_sentences']
    )


def run_single_example(example, selected_arm, kg_arm, reward_fn, validator,
                       config, use_safety=True):
    """
    Run one example through the pipeline with a given arm selection.
    Returns dict with all metrics.
    """
    question = example['QUESTION']
    contexts = example['CONTEXTS']
    gold_answer = example['final_decision']
    long_answer = " ".join(example.get('LONG_ANSWER', example.get('long_answer', [])))

    # Retrieve
    t0 = time.time()
    if selected_arm == 0:
        retrieved = retrieve_fast(question, contexts,
                                  top_k=config['retrieval']['fast_arm']['top_k'])
    elif selected_arm == 1:
        retrieved = retrieve_deep(question, contexts,
                                  top_k=config['retrieval']['deep_arm']['top_k'])
    else:
        retrieved = retrieve_kg(question, contexts,
                                top_k=config['retrieval']['kg_arm']['top_k'],
                                kg_arm=kg_arm)
    retrieval_time = time.time() - t0

    # LLM
    t0 = time.time()
    predicted = answer_question(question, retrieved,
                                max_new_tokens=config['llm']['max_new_tokens'])
    llm_time = time.time() - t0

    # Safety
    if use_safety:
        is_safe, reason, details = validator.validate(
            question=question, retrieved_context=retrieved,
            predicted_answer=predicted, confidence=None
        )
        if not is_safe:
            predicted = "abstain"
    else:
        is_safe = True

    total_time = retrieval_time + llm_time
    correct = (predicted == gold_answer)

    # Reward
    reward, components = reward_fn.compute_reward(
        predicted_answer=predicted,
        gold_answer=gold_answer,
        generated_response=" ".join(retrieved),
        reference_text=long_answer,
        time_taken=total_time,
        safety_passed=is_safe,
    )

    return {
        'predicted': predicted,
        'gold': gold_answer,
        'correct': correct,
        'reward': reward,
        'components': components,
        'retrieval_time': retrieval_time,
        'llm_time': llm_time,
        'total_time': total_time,
        'is_safe': is_safe,
        'arm': selected_arm,
    }


def run_strategy(strategy_name, examples, config, kg_arm, reward_fn, validator,
                 bandit=None, use_safety=True):
    """
    Run a full strategy over all examples.
    strategy_name: always_fast, always_deep, always_graph, random, bandit, oracle
    """
    results = []
    correct_count = 0

    for i, example in enumerate(examples):
        question = example['QUESTION']
        contexts = example['CONTEXTS']

        # Arm selection
        if strategy_name == 'always_fast':
            arm = 0
        elif strategy_name == 'always_deep':
            arm = 1
        elif strategy_name == 'always_graph':
            arm = 2
        elif strategy_name == 'random':
            arm = np.random.choice([0, 1, 2])
        elif strategy_name == 'bandit':
            ctx = extract_context(question, contexts, bandit=bandit, kg_arm=kg_arm)
            arm, probs, ucb = bandit.select_arm_with_probs(ctx)
        elif strategy_name == 'oracle':
            # Try all three arms, pick whichever gets it right
            best = None
            for try_arm in [0, 1, 2]:
                res = run_single_example(example, try_arm, kg_arm, reward_fn,
                                         validator, config, use_safety)
                if best is None or res['reward'] > best['reward']:
                    best = res
            results.append(best)
            if best['correct']:
                correct_count += 1
            if (i + 1) % 50 == 0:
                print(f"  [{strategy_name}] {i+1}/{len(examples)} "
                      f"acc={correct_count/(i+1):.1%}")
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        res = run_single_example(example, arm, kg_arm, reward_fn, validator,
                                 config, use_safety)

        # Update bandit if applicable
        if strategy_name == 'bandit' and bandit is not None:
            bandit.update(arm, ctx, res['reward'])

        results.append(res)
        if res['correct']:
            correct_count += 1

        if (i + 1) % 50 == 0:
            print(f"  [{strategy_name}] {i+1}/{len(examples)} "
                  f"acc={correct_count/(i+1):.1%}")

    return results


def compute_metrics(results):
    """Compute summary metrics from a list of per-example results."""
    n = len(results)
    if n == 0:
        return {}

    correct = sum(r['correct'] for r in results)
    rewards = [r['reward'] for r in results]
    latencies = [r['total_time'] for r in results]
    guideline = [r['components']['r_guideline'] for r in results]

    return {
        'n': n,
        'accuracy': correct / n,
        'correct': correct,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_latency': np.mean(latencies),
        'mean_guideline': np.mean(guideline),
        'rewards': rewards,
        'latencies': latencies,
    }


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    means = []
    data = np.array(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def cohens_d(x, y):
    """Effect size (Cohen's d) between two samples."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) +
                          (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (mx - my) / pooled_std


def statistical_tests(bandit_results, baseline_results, baseline_name,
                      n_comparisons=5):
    """
    Paired t-test with Bonferroni correction, bootstrap CI, Cohen's d.
    """
    bandit_rewards = bandit_results['rewards']
    base_rewards = baseline_results['rewards']

    n = min(len(bandit_rewards), len(base_rewards))
    br = bandit_rewards[:n]
    bsr = base_rewards[:n]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(br, bsr)
    # Bonferroni correction
    adjusted_p = min(p_value * n_comparisons, 1.0)

    # Bootstrap CI on reward difference
    diffs = [a - b for a, b in zip(br, bsr)]
    ci_lower, ci_upper = bootstrap_ci(diffs)

    # Effect size
    d = cohens_d(br, bsr)

    return {
        'baseline': baseline_name,
        't_stat': round(t_stat, 4),
        'p_value': round(p_value, 6),
        'p_adjusted': round(adjusted_p, 6),
        'significant': adjusted_p < 0.05,
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'cohens_d': round(d, 4),
        'bandit_mean': round(np.mean(br), 4),
        'baseline_mean': round(np.mean(bsr), 4),
    }


def confusion_matrix(results):
    """Build confusion matrix for yes/no/maybe predictions."""
    labels = ['yes', 'no', 'maybe']
    matrix = {g: {p: 0 for p in labels + ['abstain']} for g in labels}
    for r in results:
        gold = r['gold']
        pred = r['predicted']
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def error_analysis(results):
    """Categorise errors by gold answer type."""
    gold_dist = Counter(r['gold'] for r in results)
    pred_dist = Counter(r['predicted'] for r in results)

    errors_by_gold = {}
    for label in ['yes', 'no', 'maybe']:
        subset = [r for r in results if r['gold'] == label]
        total = len(subset)
        correct = sum(r['correct'] for r in subset)
        errors_by_gold[label] = {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
        }

    return {
        'gold_distribution': dict(gold_dist),
        'prediction_distribution': dict(pred_dist),
        'accuracy_by_gold': errors_by_gold,
        'confusion_matrix': confusion_matrix(results),
    }


def run_ablations(examples, config, kg_arm, reward_fn, validator):
    """
    Ablation studies:
    1. No safety validator
    2. Random arm selection (no bandit)
    3. 2-arm only (no KG arm)
    4. Remove each reward component
    """
    print("\nABLATION STUDIES\n")
    ablation_results = {}

    # Full system baseline
    print("Running full system...")
    bandit_full = LinUCB(
        n_arms=config['bandit']['n_arms'],
        n_features=config['bandit']['n_features'],
        alpha=config['bandit']['alpha']
    )
    full_res = run_strategy('bandit', examples, config, kg_arm, reward_fn,
                            validator, bandit=bandit_full, use_safety=True)
    ablation_results['full_system'] = compute_metrics(full_res)
    print(f"  Full system: {ablation_results['full_system']['accuracy']:.1%}")

    # Ablation 1: No safety
    print("Running without safety validator...")
    bandit_ns = LinUCB(
        n_arms=config['bandit']['n_arms'],
        n_features=config['bandit']['n_features'],
        alpha=config['bandit']['alpha']
    )
    ns_res = run_strategy('bandit', examples, config, kg_arm, reward_fn,
                          validator, bandit=bandit_ns, use_safety=False)
    ablation_results['no_safety'] = compute_metrics(ns_res)
    print(f"  No safety: {ablation_results['no_safety']['accuracy']:.1%}")

    # Ablation 2: Random selection (no bandit learning)
    print("Running with random arm selection...")
    rand_res = run_strategy('random', examples, config, kg_arm, reward_fn,
                            validator, use_safety=True)
    ablation_results['random_selection'] = compute_metrics(rand_res)
    print(f"  Random: {ablation_results['random_selection']['accuracy']:.1%}")

    # Ablation 3: 2-arm only (fast + deep, no graph)
    print("Running 2-arm bandit (no KG arm)...")
    bandit_2arm = LinUCB(n_arms=2, n_features=config['bandit']['n_features'],
                         alpha=config['bandit']['alpha'])
    two_arm_res = []
    correct_2 = 0
    for i, ex in enumerate(examples):
        q = ex['QUESTION']
        c = ex['CONTEXTS']
        ctx = extract_context(q, c, bandit=bandit_2arm, kg_arm=kg_arm)
        # Only consider arms 0 and 1
        ctx_2 = ctx[:config['bandit']['n_features']]
        arm, _, _ = bandit_2arm.select_arm_with_probs(ctx_2)
        arm = min(arm, 1)  # clamp to 2 arms
        res = run_single_example(ex, arm, kg_arm, reward_fn, validator,
                                 config, use_safety=True)
        bandit_2arm.update(arm, ctx_2, res['reward'])
        two_arm_res.append(res)
        if res['correct']:
            correct_2 += 1
        if (i + 1) % 50 == 0:
            print(f"  [2-arm] {i+1}/{len(examples)} acc={correct_2/(i+1):.1%}")
    ablation_results['two_arm'] = compute_metrics(two_arm_res)
    print(f"  2-arm: {ablation_results['two_arm']['accuracy']:.1%}")

    return ablation_results


def print_results_table(all_metrics):
    """Print a comparison table."""
    print(f"\n{'Strategy':<20} {'Accuracy':>10} {'Mean Reward':>12} "
          f"{'Mean Latency':>13} {'Guideline':>10}")
    print("-" * 70)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['accuracy']:>9.1%} {m['mean_reward']:>12.4f} "
              f"{m['mean_latency']:>12.3f}s {m['mean_guideline']:>10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000,
                        help='Number of examples (default 1000)')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--skip-oracle', action='store_true',
                        help='Skip oracle (3x slower)')
    parser.add_argument('--skip-ablations', action='store_true',
                        help='Skip ablation studies')
    args = parser.parse_args()

    print("COMPREHENSIVE EVALUATION\n")

    # Init
    init_shared(args.config)

    # Load data
    print("Loading PubMedQA data...")
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    examples = list(data.values())[:args.n]
    print(f"Evaluating on {len(examples)} examples\n")

    # Part A+B: Run all strategies
    strategies = [
        ('always_fast', 'Always-Fast'),
        ('always_deep', 'Always-Deep'),
        ('always_graph', 'Always-Graph'),
        ('random', 'Random'),
    ]

    all_results = {}
    all_metrics = {}

    for strat_key, strat_name in strategies:
        print(f"\nRunning {strat_name}...")
        res = run_strategy(strat_key, examples, CONFIG, KG_ARM, REWARD_FN,
                           VALIDATOR, use_safety=True)
        all_results[strat_name] = res
        all_metrics[strat_name] = compute_metrics(res)
        print(f"  {strat_name}: {all_metrics[strat_name]['accuracy']:.1%} accuracy, "
              f"{all_metrics[strat_name]['mean_reward']:.4f} reward")

    # Bandit
    print("\nRunning LinUCB Bandit...")
    bandit = LinUCB(
        n_arms=CONFIG['bandit']['n_arms'],
        n_features=CONFIG['bandit']['n_features'],
        alpha=CONFIG['bandit']['alpha']
    )
    bandit_res = run_strategy('bandit', examples, CONFIG, KG_ARM, REWARD_FN,
                              VALIDATOR, bandit=bandit, use_safety=True)
    all_results['LinUCB Bandit'] = bandit_res
    all_metrics['LinUCB Bandit'] = compute_metrics(bandit_res)
    print(f"  Bandit: {all_metrics['LinUCB Bandit']['accuracy']:.1%} accuracy, "
          f"{all_metrics['LinUCB Bandit']['mean_reward']:.4f} reward")

    # Oracle (optional, slow)
    if not args.skip_oracle:
        print("\nRunning Oracle (tries all arms)...")
        oracle_res = run_strategy('oracle', examples, CONFIG, KG_ARM, REWARD_FN,
                                  VALIDATOR, use_safety=True)
        all_results['Oracle'] = oracle_res
        all_metrics['Oracle'] = compute_metrics(oracle_res)
        print(f"  Oracle: {all_metrics['Oracle']['accuracy']:.1%} accuracy")

    # Print comparison table
    print("\n\nRESULTS SUMMARY")
    print_results_table(all_metrics)

    # Part C: Statistical tests
    print("\n\nSTATISTICAL TESTS (Bandit vs each baseline)\n")
    baselines = [k for k in all_metrics if k != 'LinUCB Bandit' and k != 'Oracle']
    n_comp = len(baselines)
    stat_results = []
    for bname in baselines:
        sr = statistical_tests(all_metrics['LinUCB Bandit'], all_metrics[bname],
                               bname, n_comparisons=n_comp)
        stat_results.append(sr)
        sig = "YES" if sr['significant'] else "no"
        print(f"  vs {bname:<15} p={sr['p_adjusted']:.4f} d={sr['cohens_d']:+.3f} "
              f"sig={sig}  bandit={sr['bandit_mean']:.4f} base={sr['baseline_mean']:.4f}")

    # Part D: Ablation studies
    if not args.skip_ablations:
        ablation_metrics = run_ablations(examples, CONFIG, KG_ARM, REWARD_FN,
                                         VALIDATOR)
        print("\nABLATION RESULTS")
        print_results_table(ablation_metrics)

    # Part E: Error analysis (on bandit results)
    print("\n\nERROR ANALYSIS (Bandit)\n")
    ea = error_analysis(bandit_res)

    print("Gold answer distribution:")
    for label, count in ea['gold_distribution'].items():
        print(f"  {label}: {count}")

    print("\nPrediction distribution:")
    for label, count in ea['prediction_distribution'].items():
        print(f"  {label}: {count}")

    print("\nAccuracy by gold answer type:")
    for label, info in ea['accuracy_by_gold'].items():
        print(f"  {label}: {info['correct']}/{info['total']} = "
              f"{info['accuracy']:.1%}")

    print("\nConfusion matrix (rows=gold, cols=predicted):")
    cm = ea['confusion_matrix']
    cols = ['yes', 'no', 'maybe', 'abstain']
    print(f"  {'':>8}", end="")
    for c in cols:
        print(f"{c:>8}", end="")
    print()
    for gold in ['yes', 'no', 'maybe']:
        print(f"  {gold:>8}", end="")
        for pred in cols:
            print(f"{cm[gold].get(pred, 0):>8}", end="")
        print()

    # Part F: Arm selection over time (for plotting)
    arm_over_time = [r['arm'] for r in bandit_res]
    # Split into early vs late
    mid = len(arm_over_time) // 2
    early_arms = Counter(arm_over_time[:mid])
    late_arms = Counter(arm_over_time[mid:])
    print("\nArm selection distribution:")
    arm_names = {0: 'Fast', 1: 'Deep', 2: 'Graph'}
    print(f"  {'':>8} {'Early':>8} {'Late':>8}")
    for a in [0, 1, 2]:
        print(f"  {arm_names[a]:>8} {early_arms.get(a, 0):>8} "
              f"{late_arms.get(a, 0):>8}")

    # Cumulative regret (vs oracle if available)
    if 'Oracle' in all_metrics:
        oracle_rewards = all_metrics['Oracle']['rewards']
        bandit_rewards = all_metrics['LinUCB Bandit']['rewards']
        n = min(len(oracle_rewards), len(bandit_rewards))
        cum_regret = np.cumsum([oracle_rewards[i] - bandit_rewards[i]
                                for i in range(n)])
        print(f"\nCumulative regret (vs Oracle): {cum_regret[-1]:.2f}")
        print(f"Average regret per query: {cum_regret[-1]/n:.4f}")

    # Save everything
    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'n_examples': len(examples),
        'metrics': {},
        'statistical_tests': stat_results,
        'error_analysis': ea,
        'arm_selection_early': dict(early_arms),
        'arm_selection_late': dict(late_arms),
    }

    for name, m in all_metrics.items():
        save_data['metrics'][name] = {
            'accuracy': m['accuracy'],
            'mean_reward': m['mean_reward'],
            'std_reward': m['std_reward'],
            'mean_latency': m['mean_latency'],
            'mean_guideline': m['mean_guideline'],
            'n': m['n'],
        }

    if not args.skip_ablations:
        save_data['ablations'] = {}
        for name, m in ablation_metrics.items():
            save_data['ablations'][name] = {
                'accuracy': m['accuracy'],
                'mean_reward': m['mean_reward'],
                'mean_latency': m['mean_latency'],
            }

    with open(output_dir / "full_results.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save per-example bandit results for plotting
    bandit_detail = []
    for r in bandit_res:
        bandit_detail.append({
            'predicted': r['predicted'],
            'gold': r['gold'],
            'correct': r['correct'],
            'reward': r['reward'],
            'arm': r['arm'],
            'total_time': r['total_time'],
            'r_guideline': r['components']['r_guideline'],
            'r_quality': r['components']['r_quality'],
        })

    with open(output_dir / "bandit_per_example.json", 'w') as f:
        json.dump(bandit_detail, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print("  full_results.json       (summary + stats + error analysis)")
    print("  bandit_per_example.json (per-example for plotting)")
    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
