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
from reward.reward_function import RewardFunction, create_reward_function


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
    
    # Initialise bandit (3 arms: 0=Fast, 1=Deep, 2=Graph)
    bandit = LinUCB(
    n_arms=config['bandit']['n_arms'],
    n_features=config['bandit']['n_features'],
    alpha=config['bandit']['alpha']
    )
    print("Initialised LinUCB bandit (alpha=2.0)")

    # Initialise safety validator
    validator = SafetyValidator(
    confidence_threshold=config['safety']['confidence_threshold'],
    min_evidence_sentences=config['safety']['min_evidence_sentences']
    )
    print("Initialised Safety Validator")

    # Initialize reward function (Section 3.3.2)
    reward_fn = create_reward_function(config)
    print(f"Initialised Reward Function: {reward_fn}")
    
    # Track results
    results = {
        'examples': [],
        'cumulative_accuracy': [],
        'arm_selections': [],
        'rewards': [],
        'reward_components': []
    }
    
    correct_count = 0
    
    print("Running Pipleine\n")

    # Run pipeline on each example
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}/{len(examples)}")
        
        question = example['QUESTION']
        contexts = example['CONTEXTS']
        gold_answer = example['final_decision']
        # LONG_ANSWER serves as guideline proxy for BERTScore
        long_answer = " ".join(example.get('LONG_ANSWER', example.get('long_answer', [])))
        
        print(f"Question: {question[:80]}...")
        
        # 1. Extract context features
        context_features = extract_context(question, contexts, bandit=bandit, kg_arm=kg_arm)
        
        # 2. Bandit selects arm
        selected_arm, arm_probs, ucb_scores = bandit.select_arm_with_probs(context_features)
        arm_names = ["Fast", "Deep", "Graph"]
        arm_name = arm_names[selected_arm]
        print(f"Bandit selected: {arm_name} (prob={arm_probs[selected_arm]:.3f}, α={bandit.alpha:.4f})")
        
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
            safety_details = {}

        if not is_safe:
            print(f"  !  ABSTAINED: {safety_reason}")
            predicted_answer = "abstain"
        else:
            print(f"✓ Safety checks passed")


        print(f"Gold answer: {gold_answer}")
        
        # 6. Compute reward (4-component weighted reward)
        total_time = retrieval_time + llm_time
        
        reward, components = reward_fn.compute_reward(
            predicted_answer=predicted_answer,
            gold_answer=gold_answer,
            # generated_response=predicted_answer,  # For PubMedQA yes/no/maybe
            generated_response=" ".join(retrieved), # Retrived context for bertscore
            reference_text=long_answer,            # Expert long answer as guideline proxy
            time_taken=total_time,
            safety_passed=is_safe,
        )
        
        # Track correctness
        correct = (predicted_answer == gold_answer)
        if correct:
            correct_count += 1
        
        # Display reward breakdown
        status = '✓' if correct else ('⚠' if predicted_answer == "abstain" else '✗')
        print(f"Reward: {reward:.4f} {status}")
        print(f"  R_guideline={components['r_guideline']:.3f} "
              f"R_quality={components['r_quality']:.1f} "
              f"R_latency={components['r_latency']:.3f} "
              f"R_safety={components['r_safety']:.1f}"
              f"{' [KILL-SWITCH]' if components['kill_switch_triggered'] else ''}")
        
        # 6. Update bandit
        bandit.update(selected_arm, context_features, reward)
        
        # Track results
        results['examples'].append({
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': predicted_answer,
            'selected_arm': arm_name,
            'selected_arm_idx': selected_arm,
            'correct': correct,
            'retrieval_time': retrieval_time,
            'llm_time': llm_time,
            'total_time': total_time,
            'reward': reward,
            'reward_components': components,
            'safety_passed': is_safe,
            'safety_reason': safety_reason,
            'context_vector': context_features.tolist(),
            'arm_probabilities': arm_probs.tolist(),  # π₀(a|x) for all arms
            'ucb_scores': ucb_scores.tolist(),
            'alpha': bandit.alpha,
        })

        results['arm_selections'].append(selected_arm)
        results['rewards'].append(reward)
        results['reward_components'].append(components)
        results['cumulative_accuracy'].append(correct_count / (i + 1))
        
        # Print running accuracy
        running_acc = correct_count / (i + 1)
        print(f"Running accuracy: {running_acc:.1%} ({correct_count}/{i+1})")
    
    # Final summary
    print("FINAL RESULTS\n")
    
    final_accuracy = correct_count / len(examples)
    print(f"\nFinal Accuracy: {final_accuracy:.1%} ({correct_count}/{len(examples)})")
    
    # Arm selection statistics
    arm_counts = np.bincount(results['arm_selections'], minlength=3)
    print(f"\nArm Selection:")
    for idx, name in enumerate(["Fast", "Deep", "Graph"]):
        print(f"  {name}: {arm_counts[idx]} times ({arm_counts[idx]/len(examples):.1%})")
    
    # Accuracy per arm
    print(f"\nAccuracy by Arm:")
    for arm_name in ["Fast", "Deep", "Graph"]:
        arm_correct = sum(1 for ex in results['examples'] if ex['selected_arm'] == arm_name and ex['correct'])
        arm_total = sum(1 for ex in results['examples'] if ex['selected_arm'] == arm_name)
        if arm_total > 0:
            print(f"  {arm_name}: {arm_correct}/{arm_total} = {arm_correct/arm_total:.1%}")
    
    # Reward component averages
    print(f"\nAverage Reward Components:")
    avg_guideline = np.mean([c['r_guideline'] for c in results['reward_components']])
    avg_quality = np.mean([c['r_quality'] for c in results['reward_components']])
    avg_latency = np.mean([c['r_latency'] for c in results['reward_components']])
    avg_safety = np.mean([c['r_safety'] for c in results['reward_components']])
    avg_reward = np.mean(results['rewards'])
    kill_count = sum(1 for c in results['reward_components'] if c['kill_switch_triggered'])
    
    print(f"  R_guideline (avg): {avg_guideline:.4f}")
    print(f"  R_quality   (avg): {avg_quality:.4f}")
    print(f"  R_latency   (avg): {avg_latency:.4f}")
    print(f"  R_safety    (avg): {avg_safety:.4f}")
    print(f"  Total reward(avg): {avg_reward:.4f}")
    print(f"  Kill-switch triggered: {kill_count}/{len(examples)} times")
    
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

    # Save bandit weights for persistence across runs
    weights_path = config['data']['output_dir'] + "bandit_weights.pkl"
    bandit.save_weights(weights_path)
    print(f"Bandit weights saved to: {weights_path}")
    
    # Save off-policy log (dedicated file for IPS estimator)
    # This is the D = {(x_t, a_t, r_t, π₀(a_t|x_t))} dataset
    offpolicy_log = []
    for ex in results['examples']:
        offpolicy_log.append({
            'context_vector': ex['context_vector'],
            'selected_arm': ex['selected_arm_idx'],
            'reward': ex['reward'],
            'arm_probabilities': ex['arm_probabilities'],
        })
    
    log_path = config['data']['output_dir'] + "offpolicy_log.json"
    with open(log_path, 'w') as f:
        json.dump(offpolicy_log, f, indent=2)
    print(f"Off-policy log saved to: {log_path} ({len(offpolicy_log)} entries)")

    return results


if __name__ == "__main__":
    results = run_pipeline(config_path="configs/config.yaml")