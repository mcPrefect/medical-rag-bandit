"""
Baseline Evaluation: Compare Bandit vs Simple Strategies

Tests:
1. Random selection (baseline)
2. Always-Fast (baseline)
3. Always-Deep (baseline)
4. LinUCB Bandit (our system)
"""

import json
import time
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

# Also suppress transformers warnings
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from retrieval.fast_arm import retrieve_fast
from retrieval.deep_arm import retrieve_deep
from bandit.linucb import LinUCB, extract_context
from llm.llm_wrapper import answer_question
from reward.reward_function import RewardFunction, create_reward_function
from utils.config import load_config
from safety.validator import SafetyValidator


def run_strategy(strategy, examples, strategy_name, reward_fn, validator, config):
    """
    Run a selection strategy on examples.
    
    Args:
        strategy: 'random', 'always_fast', 'always_deep', or 'bandit'
        examples: list of PubMedQA examples
        strategy_name: name for printing
        reward_fn: RewardFunction instance
        validator: SafetyValidator instance
    
    Returns:
        dict with results
    """
    print(f"Running: {strategy_name}\n")
    
    # Initialise bandit if needed
    bandit = None
    if strategy == 'bandit':
        bandit = LinUCB(
            n_arms=config['bandit']['n_arms'],
            n_features=config['bandit']['n_features'],
            alpha=config['bandit']['alpha']
        )
    
    results = {
        'strategy': strategy_name,
        'correct': 0,
        'total': 0,
        'arm_selections': {'fast': 0, 'deep': 0, 'graph': 0},
        'latencies': [],
        'rewards': [],
        'reward_components': [],
        'predictions': []
    }
    
    for i, example in enumerate(examples):
        question = example['QUESTION']
        contexts = example['CONTEXTS']
        gold_answer = example['final_decision']
        long_answer = " ".join(example.get('LONG_ANSWER', example.get('long_answer', [])))

        # Select arm based on strategy
        if strategy == 'random':
            selected_arm = np.random.choice([0, 1])
        elif strategy == 'always_fast':
            selected_arm = 0
        elif strategy == 'always_deep':
            selected_arm = 1
        elif strategy == 'bandit':
            context_features = extract_context(question, contexts)
            selected_arm = bandit.select_arm(context_features)
        
        arm_names_map = {0: "fast", 1: "deep", 2: "graph"}
        arm_name = arm_names_map.get(selected_arm, "fast")
        results['arm_selections'][arm_name] += 1
        
        # Retrieve (for timing, but use all context for LLM)
        start = time.time()
        if selected_arm == 0:
            retrieved = retrieve_fast(question, contexts, top_k=config['retrieval']['fast_arm']['top_k'])
        elif selected_arm == 1:
            retrieved = retrieve_deep(question, contexts, top_k=config['retrieval']['deep_arm']['top_k'])
        else:
            retrieved = retrieve_fast(question, contexts, top_k=config['retrieval']['fast_arm']['top_k'])
        retrieval_time = time.time() - start
        
        # LLM answer
        start = time.time()
        # predicted = answer_question(question, retrieved, max_new_tokens=50)
        predicted = answer_question(question, retrieved, max_new_tokens=config['llm']['max_new_tokens'])
        llm_time = time.time() - start
        
        total_time = retrieval_time + llm_time
        results['latencies'].append(total_time)

        # Safety validation
        if config['safety']['enabled']:
            is_safe, safety_reason, _ = validator.validate(
                question=question,
                retrieved_context=retrieved,
                predicted_answer=predicted,
                confidence=None
            )
        else:
            is_safe = True
        
        if not is_safe:
            predicted = "abstain"
        
        # Compute reward using 4-component function
        reward, components = reward_fn.compute_reward(
            predicted_answer=predicted,
            gold_answer=gold_answer,
            generated_response=predicted,
            reference_text=long_answer,
            time_taken=total_time,
            safety_passed=is_safe,
        )
        
        # Check correctness
        correct = (predicted == gold_answer)
        if correct:
            results['correct'] += 1
        results['total'] += 1
        
        results['rewards'].append(reward)
        results['reward_components'].append(components)
        results['predictions'].append({
            'question': question[:80],
            'predicted': predicted,
            'gold': gold_answer,
            'correct': correct,
            'arm': arm_name,
            'reward': reward,
        })
        
        # Update bandit if applicable (with latency-aware reward)
        # if strategy == 'bandit':
        #     base_reward = 1.0 if correct else 0.0
        #     latency_penalty = 0.1 * total_time
        #     reward = base_reward - latency_penalty
        #     bandit.update(selected_arm, context_features, reward)
        if strategy == 'bandit':
            bandit.update(selected_arm, context_features, reward)
        
        # Progress
        if (i + 1) % 20 == 0:
            acc = results['correct'] / results['total']
            avg_r = np.mean(results['rewards'])
            # print(f"  Progress: {i+1}/{len(examples)} - Accuracy: {acc:.1%}")
            print(f"  Progress: {i+1}/{len(examples)} - Accuracy: {acc:.1%} - Avg Reward: {avg_r:.4f}")
    
    # Calculate final stats
    results['accuracy'] = results['correct'] / results['total']
    results['avg_latency'] = np.mean(results['latencies'])
    results['avg_reward'] = np.mean(results['rewards'])
       
    return results


def print_comparison(all_results):
    """Print comparison table of all strategies."""
    print("Baseline Comparison\n")
    
    print(f"\n{'Strategy':<25} {'Accuracy':<12} {'Avg Reward':<12} {'Fast%':<10} {'Deep%':<10} {'Latency':<10}")    
    
    for result in all_results:
        strategy = result['strategy']
        accuracy = result['accuracy']
        avg_reward = result['avg_reward']
        fast_pct = result['arm_selections']['fast'] / result['total']
        deep_pct = result['arm_selections']['deep'] / result['total']
        latency = result['avg_latency']
        
        print(f"{strategy:<25} {accuracy:>10.1%}  {avg_reward:>10.4f}  {fast_pct:>8.1%}  {deep_pct:>8.1%}  {latency:>8.3f}s")    
    
     # Find best by reward (the metric the bandit actually optimises)
    best_reward = max(all_results, key=lambda x: x['avg_reward'])
    best_acc = max(all_results, key=lambda x: x['accuracy'])
    
    print(f"\nBest by Reward: {best_reward['strategy']} ({best_reward['avg_reward']:.4f})")
    print(f"Best by Accuracy: {best_acc['strategy']} ({best_acc['accuracy']:.1%})")



def main():
    """Run all baseline comparisons."""
    print("Baseline Evaluation")
    
    # Load data
    print("\nLoading PubMedQA data...")
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)

    # Load config
    config = load_config("configs/config.yaml")
    
    # Use 100 examples for evaluation
    n_examples = config['experiment']['n_examples']
    examples = list(data.values())[:n_examples]
    print(f"Evaluating on {n_examples} examples\n")
    
    # Initialise reward function and safety validator
    reward_fn = create_reward_function(config)
    print(f"Reward Function: {reward_fn}")
    
    validator = SafetyValidator(
        confidence_threshold=config['safety']['confidence_threshold'],
        min_evidence_sentences=config['safety']['min_evidence_sentences']
    )

    # Run all strategies
    strategies = [
        ('random', 'Random Selection'),
        ('always_fast', 'Always-Fast'),
        ('always_deep', 'Always-Deep'),
        ('bandit', 'LinUCB Bandit (Ours)')
    ]
    
    all_results = []
    
    for strategy, name in strategies:
        result = run_strategy(strategy, examples, name, reward_fn, validator, config)
        all_results.append(result)
        
        # Print individual results
        print(f"\n{name} Results:")
        print(f"  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        print(f"  Fast selections: {result['arm_selections']['fast']}")
        print(f"  Deep selections: {result['arm_selections']['deep']}")
        print(f"  Avg latency: {result['avg_latency']:.3f}s")
    
    # Print comparison table
    print_comparison(all_results)
    
    # Save results
    output_file = "results/baseline_comparison.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'n_examples': n_examples,
        'reward_config': {
            'w_guideline': reward_fn.w_guideline,
            'w_quality': reward_fn.w_quality,
            'w_latency': reward_fn.w_latency,
            'w_safety': reward_fn.w_safety,
            'time_budget': reward_fn.time_budget,
            'safety_kill_switch': reward_fn.safety_kill_switch,
        },
        'results': [
            {
                'strategy': r['strategy'],
                'accuracy': r['accuracy'],
                'avg_reward': r['avg_reward'],
                'correct': r['correct'],
                'total': r['total'],
                'arm_selections': r['arm_selections'],
                'avg_latency': r['avg_latency'],
            }
            for r in all_results
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()