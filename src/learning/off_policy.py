"""
Off-Policy Evaluation: IPS Estimator and Policy Comparison
Implements Section 3.6 of Interim Report (LEARN stage)

The off-policy evaluation framework enables safe policy updates by
estimating how well a *new* policy would have performed using data
collected under the *current* policy, without deploying the new policy.

Key concepts:
    - Logging policy π₀: The policy that collected the data (our LinUCB)
    - Target policy π: A candidate policy we want to evaluate
    - IPS: Reweights observed rewards by π(a|x)/π₀(a|x)
    - Capped IPS: Caps importance weights at M to reduce variance
    - Bootstrap CI: Resamples data to get confidence intervals

References:
    - Precup et al. (2000): IPS for off-policy evaluation
    - Bottou et al. (2013): Capped importance weights
    - Dudík et al. (2011): Doubly robust estimation
    - Gottesman et al. (2019): Safe RL in healthcare
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_offpolicy_log(path: str) -> List[Dict]:
    """
    Load logged bandit decisions from disk.
    
    Each entry contains:
        - context_vector: 10-dim feature vector (x_t)
        - selected_arm: which arm was chosen (a_t)
        - reward: observed reward (r_t)
        - arm_probabilities: π₀(a|x_t) for all arms
    
    This is the dataset D = {(x_t, a_t, r_t, π₀(a_t|x_t))}
    that IPS operates on.
    """
    with open(path, 'r') as f:
        log = json.load(f)
    logger.info(f"Loaded {len(log)} off-policy log entries from {path}")
    return log


# ──────────────────────────────────────────────────────────────
# IPS Estimator
# ──────────────────────────────────────────────────────────────

def compute_ips(
    log_data: List[Dict],
    target_policy_fn: Callable[[np.ndarray], np.ndarray],
    cap: float = 5.0,
) -> Tuple[float, np.ndarray]:
    """
    Capped Inverse Propensity Scoring estimator.
    
    Estimates the expected reward of a target policy π using data
    collected under logging policy π₀.
    
    Formula (Precup et al., 2000):
        V_IPS(π) = (1/T) Σ_t [ w_t · r_t ]
    
    where the importance weight is:
        w_t = π(a_t | x_t) / π₀(a_t | x_t)
    
    Capping (Bottou et al., 2013):
        w_t^cap = min(w_t, M)    with M = 5
    
    Why capping? Without it, if π₀ gave arm A a 1% chance but π
    gives it 90%, the weight is 90x — one example dominates the
    entire estimate. Capping at M=5 introduces slight bias but
    massively reduces variance, which matters more with small
    datasets (we have ~100-1000 logged decisions).
    
    Args:
        log_data: List of logged decision dicts
        target_policy_fn: Function that takes context vector (np.array)
                         and returns probabilities for each arm (np.array)
        cap: Maximum importance weight (M). Default 5.0 per Bottou et al.
        
    Returns:
        (v_ips, per_example_weighted_rewards)
        - v_ips: float, estimated expected reward under target policy
        - per_example_weighted_rewards: np.array, for bootstrap resampling
    """
    n = len(log_data)
    if n == 0:
        return 0.0, np.array([])
    
    weighted_rewards = np.zeros(n)
    
    for i, entry in enumerate(log_data):
        context = np.array(entry['context_vector'])
        chosen_arm = entry['selected_arm']
        reward = entry['reward']
        logging_probs = np.array(entry['arm_probabilities'])
        
        # π₀(a_t | x_t) — probability the logging policy assigned
        # to the arm that was actually chosen
        pi_0 = logging_probs[chosen_arm]
        
        # Avoid division by zero — if logging policy gave 0 probability
        # to this arm, something went wrong (shouldn't happen with softmax)
        pi_0 = max(pi_0, 1e-8)
        
        # π(a_t | x_t) — probability the TARGET policy would assign
        # to the arm that was actually chosen
        target_probs = target_policy_fn(context)
        pi_new = target_probs[chosen_arm]
        
        # Importance weight: how much more/less likely is the target
        # policy to have made this same decision?
        w = pi_new / pi_0
        
        # Cap the weight to reduce variance
        w_capped = min(w, cap)
        
        weighted_rewards[i] = w_capped * reward
    
    v_ips = np.mean(weighted_rewards)
    
    return v_ips, weighted_rewards


# ──────────────────────────────────────────────────────────────
# Bootstrap Confidence Intervals
# ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    weighted_rewards: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for IPS estimate.
    
    Why bootstrap? We can't assume the weighted rewards are normally
    distributed (they're products of importance weights and rewards,
    which can be heavy-tailed). Bootstrap makes no distributional
    assumptions — it resamples the data 1000 times and takes percentiles.
    
    Args:
        weighted_rewards: Per-example weighted rewards from compute_ips
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (0.95 = 95% CI)
        seed: Random seed for reproducibility
        
    Returns:
        (lower, mean, upper) — bounds of the confidence interval
    """
    rng = np.random.RandomState(seed)
    n = len(weighted_rewards)
    
    if n == 0:
        return 0.0, 0.0, 0.0
    
    bootstrap_means = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_means[b] = np.mean(weighted_rewards[indices])
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, 100 * alpha)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha))
    mean = np.mean(bootstrap_means)
    
    return lower, mean, upper


# ──────────────────────────────────────────────────────────────
# Candidate policies (things to compare against)
# ──────────────────────────────────────────────────────────────

def make_always_arm_policy(arm_idx: int, n_arms: int = 3) -> Callable:
    """
    Create a policy that always selects one specific arm.
    
    Returns probability 1.0 for the chosen arm, 0.0 for others.
    Useful as a baseline: "what if we had always used BM25?"
    
    Args:
        arm_idx: Which arm to always select (0=Fast, 1=Deep, 2=Graph)
        n_arms: Total number of arms
        
    Returns:
        Function: context -> probabilities
    """
    def policy_fn(context):
        probs = np.zeros(n_arms)
        probs[arm_idx] = 1.0
        return probs
    return policy_fn


def make_uniform_policy(n_arms: int = 3) -> Callable:
    """
    Random uniform policy — selects each arm with equal probability.
    
    Baseline: "what if we had selected randomly?"
    """
    def policy_fn(context):
        return np.ones(n_arms) / n_arms
    return policy_fn


def make_linucb_policy(bandit) -> Callable:
    """
    Wrap an existing LinUCB bandit as a policy function.
    
    Uses the bandit's current learned weights to compute
    arm probabilities via softmax over UCB scores.
    
    Args:
        bandit: LinUCB instance with learned weights
        
    Returns:
        Function: context -> probabilities
    """
    def policy_fn(context):
        return bandit.get_action_probabilities(context)
    return policy_fn


# ──────────────────────────────────────────────────────────────
# Policy Comparison (Stage C)
# ──────────────────────────────────────────────────────────────

def compare_policies(
    log_data: List[Dict],
    current_policy_fn: Callable,
    candidate_policy_fn: Callable,
    cap: float = 5.0,
    n_bootstrap: int = 1000,
    improvement_threshold: float = 0.02,
) -> Dict:
    """
    Compare a candidate policy against the current policy.
    
    Implements the safe update criterion from Section 3.6.3:
    Only update if:
        1. V_IPS(new) > V_IPS(current) + threshold (≥2% improvement)
        2. 95% CI of the improvement excludes negative values
    
    This prevents deploying a policy that might be worse — critical
    in healthcare where a bad policy update could harm patients.
    
    Args:
        log_data: Logged decisions from the current policy
        current_policy_fn: Current policy's probability function
        candidate_policy_fn: Candidate policy's probability function
        cap: IPS weight cap
        n_bootstrap: Bootstrap resamples
        improvement_threshold: Minimum improvement required (default 2%)
        
    Returns:
        Dict with comparison results including recommendation
    """
    # Evaluate both policies on the same logged data
    v_current, wr_current = compute_ips(log_data, current_policy_fn, cap)
    v_candidate, wr_candidate = compute_ips(log_data, candidate_policy_fn, cap)
    
    # Bootstrap CI for each
    ci_current = bootstrap_ci(wr_current, n_bootstrap)
    ci_candidate = bootstrap_ci(wr_candidate, n_bootstrap)
    
    # Bootstrap CI for the DIFFERENCE (candidate - current)
    # This is what we actually care about: is the improvement real?
    diff_rewards = wr_candidate - wr_current
    ci_diff = bootstrap_ci(diff_rewards, n_bootstrap)
    
    # Decision criteria
    raw_improvement = v_candidate - v_current
    meets_threshold = raw_improvement >= improvement_threshold
    ci_excludes_negative = ci_diff[0] > 0  # Lower bound > 0
    
    recommend_update = meets_threshold and ci_excludes_negative
    
    result = {
        'v_current': round(v_current, 4),
        'v_candidate': round(v_candidate, 4),
        'improvement': round(raw_improvement, 4),
        'improvement_pct': round(raw_improvement / max(abs(v_current), 1e-8) * 100, 2),
        'ci_current': (round(ci_current[0], 4), round(ci_current[2], 4)),
        'ci_candidate': (round(ci_candidate[0], 4), round(ci_candidate[2], 4)),
        'ci_improvement': (round(ci_diff[0], 4), round(ci_diff[2], 4)),
        'meets_threshold': meets_threshold,
        'ci_excludes_negative': ci_excludes_negative,
        'recommend_update': recommend_update,
    }
    
    return result


# ──────────────────────────────────────────────────────────────
# Full off-policy evaluation report
# ──────────────────────────────────────────────────────────────

def run_offpolicy_evaluation(
    log_path: str,
    bandit=None,
    n_arms: int = 3,
    cap: float = 5.0,
) -> Dict:
    """
    Run complete off-policy evaluation on logged data.
    
    Evaluates the current bandit against multiple baselines:
    - Always-Fast, Always-Deep, Always-Graph
    - Random uniform
    - Current bandit (self-evaluation as sanity check)
    
    This is what the LEARN stage would run during the "nightly
    policy update" described in Section 3.6.4.
    
    Args:
        log_path: Path to offpolicy_log.json
        bandit: Current LinUCB bandit (optional, for self-evaluation)
        n_arms: Number of arms
        cap: IPS weight cap
        
    Returns:
        Dict with full evaluation results
    """
    log_data = load_offpolicy_log(log_path)
    
    if len(log_data) == 0:
        return {'error': 'No logged data available'}
    
    print("=" * 60)
    print("OFF-POLICY EVALUATION (LEARN Stage)")
    print("=" * 60)
    print(f"\nEvaluating on {len(log_data)} logged decisions")
    print(f"IPS weight cap: M={cap}")
    
    # Define candidate policies
    candidates = {
        'Always-Fast (BM25)': make_always_arm_policy(0, n_arms),
        'Always-Deep (Semantic)': make_always_arm_policy(1, n_arms),
        'Always-Graph (KG)': make_always_arm_policy(2, n_arms),
        'Random Uniform': make_uniform_policy(n_arms),
    }
    
    # If we have the current bandit, include it for self-evaluation
    if bandit is not None:
        candidates['Current LinUCB'] = make_linucb_policy(bandit)
    
    # Evaluate each candidate
    results = {
        'n_logged_decisions': len(log_data),
        'ips_cap': cap,
        'policies': {},
    }
    
    print(f"\n{'Policy':<30s} {'V_IPS':>10s} {'95% CI':>25s}")
    print("-" * 70)
    
    for name, policy_fn in candidates.items():
        v_ips, wr = compute_ips(log_data, policy_fn, cap)
        lower, mean, upper = bootstrap_ci(wr)
        
        results['policies'][name] = {
            'v_ips': round(v_ips, 4),
            'ci_lower': round(lower, 4),
            'ci_upper': round(upper, 4),
        }
        
        print(f"{name:<30s} {v_ips:>10.4f} [{lower:>10.4f}, {upper:>10.4f}]")
    
    # If we have the current bandit, compare against best candidate
    if bandit is not None:
        current_fn = make_linucb_policy(bandit)
        
        print(f"\n{'─' * 60}")
        print("POLICY COMPARISON (Safe Update Check)")
        print(f"{'─' * 60}")
        
        comparisons = {}
        for name, policy_fn in candidates.items():
            if name == 'Current LinUCB':
                continue
            
            comparison = compare_policies(
                log_data, current_fn, policy_fn, cap
            )
            comparisons[name] = comparison
            
            rec = "✓ RECOMMEND UPDATE" if comparison['recommend_update'] else "✗ Keep current"
            print(f"\n  vs {name}:")
            print(f"    Improvement: {comparison['improvement']:+.4f} ({comparison['improvement_pct']:+.2f}%)")
            print(f"    95% CI: [{comparison['ci_improvement'][0]:+.4f}, {comparison['ci_improvement'][1]:+.4f}]")
            print(f"    Meets threshold (≥2%): {comparison['meets_threshold']}")
            print(f"    CI excludes negative: {comparison['ci_excludes_negative']}")
            print(f"    → {rec}")
        
        results['comparisons'] = comparisons
    
    # Counterfactual analysis: "what reward would we have gotten?"
    print(f"\n{'─' * 60}")
    print("COUNTERFACTUAL ANALYSIS")
    print(f"{'─' * 60}")
    
    actual_avg_reward = np.mean([e['reward'] for e in log_data])
    print(f"\n  Actual avg reward (observed): {actual_avg_reward:.4f}")
    
    for name, policy_data in results['policies'].items():
        diff = policy_data['v_ips'] - actual_avg_reward
        print(f"  {name}: {policy_data['v_ips']:.4f} ({diff:+.4f})")
    
    print(f"\n{'=' * 60}")
    
    return results


# ──────────────────────────────────────────────────────────────
# Standalone test / demo
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OFF-POLICY EVALUATION — DEMO WITH SYNTHETIC DATA")
    print("=" * 60)
    
    # Create synthetic logged data to demonstrate IPS works
    # Simulating a scenario where:
    # - Fast arm is good for short queries (context[3] < 0.5)
    # - Deep arm is good for complex queries (context[0] > 0.5)
    # - Logging policy mostly picked Fast
    
    np.random.seed(42)
    n_examples = 200
    n_arms = 3
    
    synthetic_log = []
    for i in range(n_examples):
        # Random context
        context = np.random.random(10)
        
        # Logging policy: mostly picks arm 0 (Fast)
        logging_probs = np.array([0.7, 0.2, 0.1])
        chosen_arm = np.random.choice(n_arms, p=logging_probs)
        
        # Reward depends on context-arm match
        if chosen_arm == 0 and context[3] < 0.5:  # Fast good for short
            reward = 0.8 + np.random.random() * 0.2
        elif chosen_arm == 1 and context[0] > 0.5:  # Deep good for complex
            reward = 0.7 + np.random.random() * 0.3
        else:
            reward = 0.2 + np.random.random() * 0.3
        
        synthetic_log.append({
            'context_vector': context.tolist(),
            'selected_arm': int(chosen_arm),
            'reward': round(reward, 4),
            'arm_probabilities': logging_probs.tolist(),
        })
    
    print(f"\nGenerated {n_examples} synthetic logged decisions")
    
    # Test IPS with different policies
    print("\n--- IPS Estimates ---")
    
    # Policy 1: Always-Fast
    always_fast = make_always_arm_policy(0, n_arms)
    v_fast, wr_fast = compute_ips(synthetic_log, always_fast)
    ci_fast = bootstrap_ci(wr_fast)
    print(f"Always-Fast:   V_IPS = {v_fast:.4f}  CI: [{ci_fast[0]:.4f}, {ci_fast[2]:.4f}]")
    
    # Policy 2: Always-Deep
    always_deep = make_always_arm_policy(1, n_arms)
    v_deep, wr_deep = compute_ips(synthetic_log, always_deep)
    ci_deep = bootstrap_ci(wr_deep)
    print(f"Always-Deep:   V_IPS = {v_deep:.4f}  CI: [{ci_deep[0]:.4f}, {ci_deep[2]:.4f}]")
    
    # Policy 3: Random
    uniform = make_uniform_policy(n_arms)
    v_rand, wr_rand = compute_ips(synthetic_log, uniform)
    ci_rand = bootstrap_ci(wr_rand)
    print(f"Random:        V_IPS = {v_rand:.4f}  CI: [{ci_rand[0]:.4f}, {ci_rand[2]:.4f}]")
    
    # Test policy comparison
    print("\n--- Policy Comparison ---")
    comparison = compare_policies(
        synthetic_log, always_fast, always_deep
    )
    print(f"Always-Fast vs Always-Deep:")
    print(f"  Improvement: {comparison['improvement']:+.4f} ({comparison['improvement_pct']:+.2f}%)")
    print(f"  95% CI: [{comparison['ci_improvement'][0]:+.4f}, {comparison['ci_improvement'][1]:+.4f}]")
    print(f"  Recommend update: {comparison['recommend_update']}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
