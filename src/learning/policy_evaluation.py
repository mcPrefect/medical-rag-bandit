"""
Policy Evaluation: Demonstrate Learning Over Time
Implements Stage D of off-policy learning (Section 3.6)

Runs the bandit in batches to demonstrate:
1. Performance improves as experience accumulates
2. Arm selection evolves based on observed rewards
3. Counterfactual analysis shows bandit beats fixed policies
4. Off-policy evaluation validates safe updates

Usage:
    python src/learning/policy_evaluation.py

This produces:
    - results/batch_learning.json  (per-batch metrics)
    - results/cumulative_regret.json  (regret over time)
    - Console output showing learning progression
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from bandit.linucb import LinUCB, extract_context
from learning.off_policy import (
    compute_ips,
    bootstrap_ci,
    compare_policies,
    make_always_arm_policy,
    make_uniform_policy,
    make_linucb_policy,
    run_offpolicy_evaluation,
)


def run_batch_learning(
    data_path: str = "data/pubmedqa/ori_pqal.json",
    n_total: int = 500,
    batch_size: int = 100,
    output_dir: str = "results/",
):
    """
    Demonstrate bandit learning over sequential batches.
    
    The idea: split the dataset into batches of 100 examples.
    Run the bandit on batch 1, save weights. Load weights,
    run on batch 2, save again. Track how accuracy and reward
    evolve across batches to show the system learns.
    
    This simulates what would happen in deployment: the bandit
    sees queries over time, accumulates experience, and gets
    better at routing queries to the right retrieval arm.
    
    Args:
        data_path: Path to PubMedQA data
        n_total: Total examples to use
        batch_size: Examples per batch
        output_dir: Where to save results
    """
    print("BATCH LEARNING DEMONSTRATION")
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    examples = list(data.values())[:n_total]
    n_batches = len(examples) // batch_size
    print(f"\n{len(examples)} examples, {batch_size} per batch, {n_batches} batches")
    
    # Initialise fresh bandit
    bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
    
    # Track metrics across batches
    batch_results = []
    all_offpolicy_log = []
    cumulative_correct = 0
    cumulative_total = 0
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_examples = examples[start:end]
        
        print(f"BATCH {batch_idx + 1}/{n_batches} (examples {start+1}-{end})")
        print(f"  α = {bandit.alpha:.4f}, step = {bandit.t}")
        
        batch_correct = 0
        batch_rewards = []
        batch_arm_selections = [0, 0, 0]
        batch_log = []
        
        for i, example in enumerate(batch_examples):
            question = example['QUESTION']
            contexts = example['CONTEXTS']
            gold_answer = example['final_decision']
            
            # Extract features
            context_features = extract_context(question, contexts, bandit=bandit)
            
            # Select arm with probabilities
            selected_arm, arm_probs, ucb_scores = bandit.select_arm_with_probs(context_features)
            
            # Simulate reward (we don't have LLM here, so use a heuristic)
            # In real pipeline this would be the full reward function
            # For demo: simulate that Deep arm is slightly better for complex queries
            # and Fast arm is better for simple ones
            complexity = context_features[0]  # Query complexity feature
            
            # Simple simulated reward based on arm-context match
            arm_base_rewards = {
                0: 0.50,  # Fast: decent baseline
                1: 0.48 + 0.10 * complexity,  # Deep: better for complex
                2: 0.45 + 0.15 * context_features[9],  # Graph: better with KG density
            }
            
            base = arm_base_rewards[selected_arm]
            noise = np.random.normal(0, 0.1)
            correct = np.random.random() < base
            
            if correct:
                reward = 0.35 + 0.15 * (1.0 - context_features[3] * 0.1)  # quality + latency
                batch_correct += 1
                cumulative_correct += 1
            else:
                reward = 0.10 + 0.05 * (1.0 - context_features[3] * 0.1)
            
            reward = max(0.0, min(1.0, reward))
            
            # Update bandit
            bandit.update(selected_arm, context_features, reward)
            
            batch_rewards.append(reward)
            batch_arm_selections[selected_arm] += 1
            cumulative_total += 1
            
            # Log for off-policy eval
            batch_log.append({
                'context_vector': context_features.tolist(),
                'selected_arm': selected_arm,
                'reward': round(reward, 4),
                'arm_probabilities': arm_probs.tolist(),
            })
        
        all_offpolicy_log.extend(batch_log)
        
        # Batch summary
        batch_acc = batch_correct / batch_size
        batch_avg_reward = np.mean(batch_rewards)
        cumulative_acc = cumulative_correct / cumulative_total
        
        arm_names = ["Fast", "Deep", "Graph"]
        arm_dist = [f"{arm_names[a]}:{batch_arm_selections[a]}" for a in range(3)]
        
        print(f"  Batch accuracy: {batch_acc:.1%}")
        print(f"  Batch avg reward: {batch_avg_reward:.4f}")
        print(f"  Cumulative accuracy: {cumulative_acc:.1%}")
        print(f"  Arm selections: {', '.join(arm_dist)}")
        print(f"  α at end: {bandit.alpha:.4f}")
        
        batch_results.append({
            'batch': batch_idx + 1,
            'accuracy': round(batch_acc, 4),
            'avg_reward': round(batch_avg_reward, 4),
            'cumulative_accuracy': round(cumulative_acc, 4),
            'arm_selections': batch_arm_selections.copy(),
            'alpha': round(bandit.alpha, 4),
            'step': bandit.t,
        })
        
        # Save bandit weights after each batch
        weights_path = f"{output_dir}bandit_weights_batch{batch_idx+1}.pkl"
        bandit.save_weights(weights_path)
    
    # Run off-policy evaluation on all accumulated data
    print("OFF-POLICY EVALUATION ON ACCUMULATED DATA")
    
    current_policy = make_linucb_policy(bandit)
    
    # Evaluate counterfactual policies
    policies = {
        'Current LinUCB': current_policy,
        'Always-Fast': make_always_arm_policy(0, 3),
        'Always-Deep': make_always_arm_policy(1, 3),
        'Always-Graph': make_always_arm_policy(2, 3),
        'Random': make_uniform_policy(3),
    }
    
    print(f"\n{'Policy':<25s} {'V_IPS':>10s} {'95% CI':>25s}")
    
    ips_results = {}
    for name, policy_fn in policies.items():
        v_ips, wr = compute_ips(all_offpolicy_log, policy_fn)
        lower, mean, upper = bootstrap_ci(wr)
        ips_results[name] = {
            'v_ips': round(v_ips, 4),
            'ci': (round(lower, 4), round(upper, 4)),
        }
        print(f"{name:<25s} {v_ips:>10.4f} [{lower:>10.4f}, {upper:>10.4f}]")
    
    # Counterfactual comparison
    print(f"\nCounterfactual: 'What if we had used a different strategy?'")
    actual_reward = np.mean([e['reward'] for e in all_offpolicy_log])
    print(f"  Actual observed avg reward: {actual_reward:.4f}")
    for name, data in ips_results.items():
        diff = data['v_ips'] - actual_reward
        print(f"  {name}: {data['v_ips']:.4f} ({diff:+.4f})")
    
    # Compute cumulative regret
    print("CUMULATIVE REGRET")
    
    # Regret = difference between oracle (best possible) and actual
    # For each example, oracle reward = max reward any arm could give
    # We approximate: oracle = best arm's avg reward
    best_policy_reward = max(r['v_ips'] for r in ips_results.values())
    
    rewards = [e['reward'] for e in all_offpolicy_log]
    cumulative_regret = []
    running_regret = 0.0
    for r in rewards:
        running_regret += (best_policy_reward - r)
        cumulative_regret.append(round(running_regret, 4))
    
    print(f"  Best policy V_IPS: {best_policy_reward:.4f}")
    print(f"  Total regret: {cumulative_regret[-1]:.4f}")
    print(f"  Avg regret per step: {cumulative_regret[-1] / len(rewards):.4f}")
    
    # Save everything
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output = {
        'batch_results': batch_results,
        'ips_results': ips_results,
        'cumulative_regret': cumulative_regret,
        'n_total': n_total,
        'batch_size': batch_size,
        'n_batches': n_batches,
    }
    
    output_path = f"{output_dir}batch_learning.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Save the full off-policy log
    log_path = f"{output_dir}offpolicy_log.json"
    with open(log_path, 'w') as f:
        json.dump(all_offpolicy_log, f, indent=2)
    print(f"Off-policy log saved to: {log_path}")
    
    # Final bandit weights
    final_weights = f"{output_dir}bandit_weights.pkl"
    bandit.save_weights(final_weights)
    
    print("LEARNING DEMONSTRATION COMPLETE")
    
    return output


if __name__ == "__main__":
    results = run_batch_learning()
