"""
Main pipeline: Bandit-driven RAG system
Connects: Bandit -> Retrieval Arms -> LLM -> Reward -> Learning
"""

import json
import time
import numpy as np
from pathlib import Path

# Import our components
import sys
sys.path.append(str(Path(__file__).parent))

from retrieval.fast_arm import retrieve_fast
from retrieval.deep_arm import retrieve_deep
from bandit.linucb import LinUCB, extract_context
from llm.llm_wrapper import answer_question


def run_pipeline(n_examples=10, output_file="results/learning_curve.json"):
    """
    Run the full bandit-driven RAG pipeline.
    n_examples: how many PubMedQA examples to run
    """
    print("BANDIT MEDICAL RAG SYSTEM\n")
    
    # Load data
    print("\nLoading PubMedQA data...")
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    
    examples = list(data.values())[:n_examples]
    print(f"Loaded {len(examples)} examples")
    
    # Initialise bandit (2 arms: 0=Fast, 1=Deep)
    bandit = LinUCB(n_arms=2, n_features=4, alpha=1.0)
    print("Initialised LinUCB bandit (alpha=1.0)")
    
    # Track results
    results = {
        'examples': [],
        'cumulative_accuracy': [],
        'arm_selections': [],
        'rewards': []
    }
    
    correct_count = 0
    
    print("Running Pipleine\n")

    # Run pipeline on each example
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}/{len(examples)}")
        
        question = example['QUESTION']
        contexts = example['CONTEXTS']
        gold_answer = example['final_decision']
        
        print(f"Question: {question[:80]}...")
        
        # 1. Extract context features
        context_features = extract_context(question, contexts)
        
        # 2. Bandit selects arm
        selected_arm = bandit.select_arm(context_features)
        arm_name = "Fast" if selected_arm == 0 else "Deep"
        print(f"Bandit selected: {arm_name}")
        
        # 3. Retrieve with selected arm (for latency difference)
        start_time = time.time()
        if selected_arm == 0:
            # Fast arm: top-3 BM25 (fast retrieval)
            retrieved = retrieve_fast(question, contexts, top_k=3)
        else:
            # Deep arm: top-5 semantic (slower but better ranking)
            retrieved = retrieve_deep(question, contexts, top_k=5)
        retrieval_time = time.time() - start_time
        
        print(f"Retrieved {len(retrieved)} sentences in {retrieval_time:.2f}s")
        
        # 4. LLM answers question
        # For MVP: Give LLM ALL context to maximise accuracy
        # The bandit still learns latency trade-offs from retrieval step
        start_time = time.time()
        predicted_answer = answer_question(question, contexts, max_new_tokens=50)
        llm_time = time.time() - start_time
        
        print(f"LLM prediction: {predicted_answer} (in {llm_time:.2f}s)")
        print(f"Gold answer: {gold_answer}")
        
        # 5. Calculate reward
        correct = (predicted_answer == gold_answer)
        
        # Simple reward: 1 if correct, 0 if wrong
        # (We can add latency penalty later)
        reward = 1.0 if correct else 0.0
        
        if correct:
            correct_count += 1
        
        print(f"Reward: {reward} {'✓' if correct else '✗'}")
        
        # 6. Update bandit
        bandit.update(selected_arm, context_features, reward)
        
        # Track results
        results['examples'].append({
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': predicted_answer,
            'selected_arm': arm_name,
            'correct': correct,
            'retrieval_time': retrieval_time,
            'llm_time': llm_time
        })
        
        results['arm_selections'].append(selected_arm)
        results['rewards'].append(reward)
        results['cumulative_accuracy'].append(correct_count / (i + 1))
        
        # Print running accuracy
        running_acc = correct_count / (i + 1)
        print(f"Running accuracy: {running_acc:.1%} ({correct_count}/{i+1})")
    
    # Final summary
    print("FINAL RESULTS\n")
    
    final_accuracy = correct_count / len(examples)
    print(f"\nFinal Accuracy: {final_accuracy:.1%} ({correct_count}/{len(examples)})")
    
    # Arm selection statistics
    arm_counts = np.bincount(results['arm_selections'])
    print(f"\nArm Selection:")
    print(f"  Fast: {arm_counts[0]} times ({arm_counts[0]/len(examples):.1%})")
    print(f"  Deep: {arm_counts[1]} times ({arm_counts[1]/len(examples):.1%})")
    
    # Accuracy per arm
    fast_correct = sum(1 for ex in results['examples'] if ex['selected_arm'] == 'Fast' and ex['correct'])
    fast_total = sum(1 for ex in results['examples'] if ex['selected_arm'] == 'Fast')
    deep_correct = sum(1 for ex in results['examples'] if ex['selected_arm'] == 'Deep' and ex['correct'])
    deep_total = sum(1 for ex in results['examples'] if ex['selected_arm'] == 'Deep')
    
    if fast_total > 0:
        print(f"\nAccuracy by Arm:")
        print(f"  Fast: {fast_correct}/{fast_total} = {fast_correct/fast_total:.1%}")
    if deep_total > 0:
        print(f"  Deep: {deep_correct}/{deep_total} = {deep_correct/deep_total:.1%}")
    
    # Average latency
    avg_retrieval = np.mean([ex['retrieval_time'] for ex in results['examples']])
    avg_llm = np.mean([ex['llm_time'] for ex in results['examples']])
    print(f"\nAverage Latency:")
    print(f"  Retrieval: {avg_retrieval:.2f}s")
    print(f"  LLM: {avg_llm:.2f}s")
    print(f"  Total: {avg_retrieval + avg_llm:.2f}s per question")
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    return results


if __name__ == "__main__":
    # Run on 10 examples for quick test
    results = run_pipeline(n_examples=100)