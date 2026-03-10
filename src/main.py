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

import warnings
warnings.filterwarnings("ignore")

# Also suppress transformers warnings
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from retrieval.fast_arm import retrieve_fast
from retrieval.deep_arm import retrieve_deep
from bandit.linucb import LinUCB, extract_context
from llm.llm_wrapper import answer_question
from safety.validator import SafetyValidator
from utils.config import load_config
from retrieval.kg_arm import KnowledgeGraphArm, retrieve_kg


def run_pipeline(config_path="configs/config.yaml"):
    # Load config
    config = load_config(config_path)
    n_examples = config['experiment']['n_examples']
    output_file = config['data']['output_dir'] + "learning_curve.json"
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

    # Initialise Knowledge Graph Arm
    print("Initialising Knowledge Graph Arm...")
    kg_arm = KnowledgeGraphArm()
    
    # Initialise bandit (2 arms: 0=Fast, 1=Deep)
    bandit = LinUCB(
    n_arms=config['bandit']['n_arms'],
    n_features=config['bandit']['n_features'],
    alpha=config['bandit']['alpha']
    )
    print("Initialised LinUCB bandit (alpha=1.0)")

    # Initialize safety validator
    validator = SafetyValidator(
    confidence_threshold=config['safety']['confidence_threshold'],
    min_evidence_sentences=config['safety']['min_evidence_sentences']
    )
    print("Initialized Safety Validator")
    
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
        arm_names = ["Fast", "Deep", "Graph"]
        arm_name = arm_names[selected_arm]
        print(f"Bandit selected: {arm_name}")
        
        # 3. Retrieve with selected arm (for latency difference)
        start_time = time.time()
        if selected_arm == 0:
            # Fast arm: top-3 BM25 (fast retrieval)
            retrieved = retrieve_fast(question, contexts, top_k=config['retrieval']['fast_arm']['top_k'])
        elif selected_arm == 1:
            # Deep arm: top-5 semantic (slower but better ranking)
            retrieved = retrieve_deep(question, contexts, top_k=config['retrieval']['deep_arm']['top_k'])
        else:
            retrieved = retrieve_kg(question, contexts, top_k=config['retrieval']['kg_arm']['top_k'], kg_arm=kg_arm)
        retrieval_time = time.time() - start_time
        
        print(f"Retrieved {len(retrieved)} sentences in {retrieval_time:.2f}s")
        
        # 4. LLM answers question
        # For MVP: Give LLM ALL context to maximise accuracy
        # The bandit still learns latency trade-offs from retrieval step
        start_time = time.time()
        predicted_answer = answer_question(question, retrieved, max_new_tokens=config['llm']['max_new_tokens'])
        llm_time = time.time() - start_time
        
        print(f"LLM prediction: {predicted_answer} (in {llm_time:.2f}s)")

        # Safety validation
        if config['safety']['enabled']:
            is_safe, safety_reason, safety_details = validator.validate(
                question=question,
                retrieved_context=retrieved,
                predicted_answer=predicted_answer,
                confidence=None
            )
        else:
            is_safe = True
            safety_reason = "Safety disabled"

        if not is_safe:
            print(f"  !  ABSTAINED: {safety_reason}")
            predicted_answer = "abstain"
        else:
            print(f"✓ Safety checks passed")


        print(f"Gold answer: {gold_answer}")
        
        # 5. Calculate reward with latency penalty
        # Handle abstentions
        if predicted_answer == "abstain":
            correct = False
            base_reward = 0.0
        else:
            correct = (predicted_answer == gold_answer)
            base_reward = 1.0 if correct else 0.0
        
        total_time = retrieval_time + llm_time
        latency_penalty = 0.1 * total_time
        reward = base_reward - latency_penalty

        # Reward = correctness - latency penalty
        # This makes bandit optimize for speed AND accuracy
        base_reward = 1.0 if correct else 0.0
        latency_penalty = config['reward']['latency_penalty_weight'] * total_time
        reward = base_reward - latency_penalty
        
        if correct:
            correct_count += 1
        
        status = '✓' if correct else ('!' if predicted_answer == "abstain" else '✗')
        print(f"Reward: {reward:.2f} {status}")
        
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
    print(f"  Graph: {arm_counts[2]} times ({arm_counts[2]/len(examples):.1%})")
    
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
    results = run_pipeline(config_path="configs/config.yaml")