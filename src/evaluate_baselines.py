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


def run_strategy(strategy, examples, strategy_name):
    """
    Run a selection strategy on examples.
    
    Args:
        strategy: 'random', 'always_fast', 'always_deep', or 'bandit'
        examples: list of PubMedQA examples
        strategy_name: name for printing
    
    Returns:
        dict with results
    """
    print(f"Running: {strategy_name}\n")
    
    # Initialize bandit if needed
    bandit = None
    if strategy == 'bandit':
        bandit = LinUCB(n_arms=2, n_features=4, alpha=1.0)
    
    results = {
        'strategy': strategy_name,
        'correct': 0,
        'total': 0,
        'arm_selections': {'fast': 0, 'deep': 0},
        'latencies': [],
        'predictions': []
    }
    
    for i, example in enumerate(examples):
        question = example['QUESTION']
        contexts = example['CONTEXTS']
        gold_answer = example['final_decision']
        
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
        
        arm_name = "fast" if selected_arm == 0 else "deep"
        results['arm_selections'][arm_name] += 1
        
        # Retrieve (for timing, but use all context for LLM)
        start = time.time()
        if selected_arm == 0:
            retrieved = retrieve_fast(question, contexts, top_k=3)
        else:
            retrieved = retrieve_deep(question, contexts, top_k=5)
        retrieval_time = time.time() - start
        
        # LLM answer
        start = time.time()
        predicted = answer_question(question, retrieved, max_new_tokens=50)
        llm_time = time.time() - start
        
        total_time = retrieval_time + llm_time
        results['latencies'].append(total_time)
        
        # Check correctness
        correct = (predicted == gold_answer)
        if correct:
            results['correct'] += 1
        results['total'] += 1
        
        results['predictions'].append({
            'question': question[:80],
            'predicted': predicted,
            'gold': gold_answer,
            'correct': correct,
            'arm': arm_name
        })
        
        # Update bandit if applicable (with latency-aware reward)
        if strategy == 'bandit':
            base_reward = 1.0 if correct else 0.0
            latency_penalty = 0.1 * total_time
            reward = base_reward - latency_penalty
            bandit.update(selected_arm, context_features, reward)
        
        # Progress
        if (i + 1) % 20 == 0:
            acc = results['correct'] / results['total']
            print(f"  Progress: {i+1}/{len(examples)} - Accuracy: {acc:.1%}")
    
    # Calculate final stats
    results['accuracy'] = results['correct'] / results['total']
    results['avg_latency'] = np.mean(results['latencies'])
    
    return results


def print_comparison(all_results):
    """Print comparison table of all strategies."""
    print("Baseline Comparison\n")
    
    print(f"\n{'Strategy':<20} {'Accuracy':<12} {'Fast%':<12} {'Deep%':<12} {'Avg Latency':<12}")
    print("-" * 80)
    
    for result in all_results:
        strategy = result['strategy']
        accuracy = result['accuracy']
        fast_pct = result['arm_selections']['fast'] / result['total']
        deep_pct = result['arm_selections']['deep'] / result['total']
        latency = result['avg_latency']
        
        print(f"{strategy:<20} {accuracy:>10.1%}  {fast_pct:>10.1%}  {deep_pct:>10.1%}  {latency:>10.3f}s")
    
    # Find best
    best = max(all_results, key=lambda x: x['accuracy'])

    print(f"Best Strategy: {best['strategy']} with {best['accuracy']:.1%} accuracy\n")



def main():
    """Run all baseline comparisons."""
    print("Baseline Evaluation")
    
    # Load data
    print("\nLoading PubMedQA data...")
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    
    # Use 100 examples for evaluation
    n_examples = 500
    examples = list(data.values())[:n_examples]
    print(f"Evaluating on {n_examples} examples\n")
    
    # Run all strategies
    strategies = [
        ('random', 'Random Selection'),
        ('always_fast', 'Always-Fast'),
        ('always_deep', 'Always-Deep'),
        ('bandit', 'LinUCB Bandit (Ours)')
    ]
    
    all_results = []
    
    for strategy, name in strategies:
        result = run_strategy(strategy, examples, name)
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
        'results': [
            {
                'strategy': r['strategy'],
                'accuracy': r['accuracy'],
                'correct': r['correct'],
                'total': r['total'],
                'arm_selections': r['arm_selections'],
                'avg_latency': r['avg_latency']
            }
            for r in all_results
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()