"""
MedQA Evaluation: Test system on USMLE-style questions
using MedRAG textbook corpus for retrieval.
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path
import sys
import faiss

sys.path.append(str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from bandit.linucb import LinUCB, extract_context
from bandit.thompson_sampling import ThompsonSampling
from llm.llm_wrapper import get_llm
from safety.validator import SafetyValidator
from reward.reward_function import RewardFunction, create_reward_function
from retrieval.kg_arm import KnowledgeGraphArm, retrieve_kg
from utils.config import load_config

import torch


# Corpus retrieval

CORPUS_CHUNKS = None
CORPUS_INDEX  = None
DEEP_MODEL    = None


def load_corpus(config):
    global CORPUS_CHUNKS, CORPUS_INDEX, DEEP_MODEL, BM25_INDEX

    print("Loading MedRAG textbook corpus...")
    with open("data/medrag/textbooks_chunks.json") as f:
        data = json.load(f)
    CORPUS_CHUNKS = data["texts"]
    print(f"  {len(CORPUS_CHUNKS):,} chunks loaded")

    print("Loading FAISS index...")
    CORPUS_INDEX = faiss.read_index("data/medrag/textbooks_index.faiss")
    print(f"  {CORPUS_INDEX.ntotal:,} vectors")

    print("Loading sentence transformer...")
    DEEP_MODEL = SentenceTransformer(
        config['retrieval']['deep_arm']['model_name'], device='cuda'
    )

def retrieve_fast_corpus(question, top_k=5):
    """BM25 over FAISS pre-filtered candidates for speed."""
    # First get top-200 candidates via FAISS (fast)
    query_emb = DEEP_MODEL.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    _, indices = CORPUS_INDEX.search(query_emb, 200)
    candidates = [(i, CORPUS_CHUNKS[i]) for i in indices[0]]
    
    # Then re-rank with BM25 over just those 200
    tokens = question.lower().split()
    candidate_texts = [c[1].lower().split() for c in candidates]
    bm25 = BM25Okapi(candidate_texts)
    scores = bm25.get_scores(tokens)
    top_local = np.argsort(scores)[::-1][:top_k]
    return [candidates[i][1] for i in top_local]

def retrieve_deep_corpus(question, top_k=5):
    query_emb = DEEP_MODEL.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    _, indices = CORPUS_INDEX.search(query_emb, top_k)
    return [CORPUS_CHUNKS[i] for i in indices[0]]


def retrieve_kg_corpus(question, top_k=5, kg_arm=None):
    if kg_arm is None:
        return retrieve_deep_corpus(question, top_k)
    return retrieve_kg(question, retrieve_deep_corpus(question, top_k=20),
                       top_k=top_k, kg_arm=kg_arm)


# Answer generation 

def answer_medqa(question, options, retrieved_context, max_new_tokens=10):
    """Generate A/B/C/D answer with confidence."""
    model, tokenizer = get_llm()

    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    context_text = "\n".join(retrieved_context)

    system_msg = (
        "You are a medical expert answering a USMLE-style question. "
        "You will be given relevant medical context and four answer options. "
        "Select the single best answer. "
        "Respond with a single letter only: A, B, C, or D."
    )

    user_msg = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    # Build token ID sets for A/B/C/D
    answer_ids = {}
    for letter in ["A", "B", "C", "D"]:
        ids = set()
        for form in [letter, letter.lower(), f" {letter}", f" {letter.lower()}"]:
            enc = tokenizer.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        answer_ids[letter] = ids

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated  = output.sequences[0, input_len:]
    answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip().upper()

    probs = torch.softmax(output.scores[0][0].float(), dim=-1)

    def cat_prob(ids):
        return probs[list(ids)].sum().item() if ids else 0.0

    letter_probs = {k: cat_prob(v) for k, v in answer_ids.items()}

    # Parse answer
    first = answer_text[0] if answer_text else ""
    if first in answer_ids:
        answer = first
    else:
        answer = max(letter_probs, key=letter_probs.get)

    return answer, letter_probs[answer]


# Single example
def run_single_medqa(example, selected_arm, kg_arm, reward_fn,
                     validator, config, use_safety=True):
    question    = example['question']
    options     = example['options']
    gold_letter = example['answer_idx']

    t0 = time.time()
    if selected_arm == 0:
        retrieved = retrieve_fast_corpus(
            question, top_k=config['retrieval']['fast_arm']['top_k']
        )
    elif selected_arm == 1:
        retrieved = retrieve_deep_corpus(
            question, top_k=config['retrieval']['deep_arm']['top_k']
        )
    else:
        retrieved = retrieve_kg_corpus(
            question, top_k=config['retrieval']['kg_arm']['top_k'],
            kg_arm=kg_arm
        )
    retrieval_time = time.time() - t0

    t0 = time.time()
    predicted, confidence = answer_medqa(question, options, retrieved)
    llm_time = time.time() - t0

    if use_safety:
        is_safe, reason, details = validator.validate(
            question=question,
            retrieved_context=retrieved,
            predicted_answer=predicted,
            confidence=confidence,
        )
        if not is_safe:
            predicted = "abstain"
    else:
        is_safe = True
        reason  = ""

    total_time = retrieval_time + llm_time
    correct    = (predicted == gold_letter)

    # Use quality component only for reward as no gold long answer available
    reward, components = reward_fn.compute_reward(
        predicted_answer="yes" if correct else "no",
        gold_answer="yes",
        generated_response=" ".join(retrieved),
        reference_text="",
        time_taken=total_time,
        safety_passed=is_safe,
    )

    return {
        'predicted':      predicted,
        'gold':           gold_letter,
        'correct':        correct,
        'reward':         reward,
        'components':     components,
        'retrieval_time': retrieval_time,
        'llm_time':       llm_time,
        'total_time':     total_time,
        'is_safe':        is_safe,
        'confidence':     confidence,
        'arm':            selected_arm,
    }


# Strategy runner

def run_strategy(strategy_name, examples, config, kg_arm,
                 reward_fn, validator, bandit=None):
    results       = []
    correct_count = 0

    for i, example in enumerate(examples):
        question = example['question']
        contexts = [f"{k}. {v}" for k, v in example['options'].items()]

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
            arm, probs, _ = bandit.select_arm_with_probs(ctx)
        elif strategy_name == 'thompson':
            ctx = extract_context(question, contexts, bandit=bandit, kg_arm=kg_arm)
            arm, probs, _ = bandit.select_arm_with_probs(ctx)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        res = run_single_medqa(
            example, arm, kg_arm, reward_fn, validator, config
        )

        if strategy_name in ('bandit', 'thompson') and bandit is not None:
            bandit.update(arm, ctx, res['reward'])

        results.append(res)
        if res['correct']:
            correct_count += 1

        if (i + 1) % 50 == 0:
            print(f"  [{strategy_name}] {i+1}/{len(examples)} "
                  f"acc={correct_count/(i+1):.1%}")

    return results


def compute_metrics(results):
    n       = len(results)
    correct = sum(r['correct'] for r in results)
    rewards = [r['reward'] for r in results]
    return {
        'n':           n,
        'accuracy':    correct / n,
        'correct':     correct,
        'mean_reward': float(np.mean(rewards)),
        'rewards':     rewards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500,
                        help='Number of MedQA test examples (max 1273)')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config['experiment']['random_seed'])

    load_corpus(config)

    print("\nLoading KG arm...")
    kg_arm = KnowledgeGraphArm(
        model_path=config['retrieval']['kg_arm']['model_path'],
        graph_path=config['retrieval']['kg_arm']['graph_path'],
        concepts_path=config['retrieval']['kg_arm']['concepts_path'],
        device=config['retrieval']['kg_arm']['device'],
    )

    reward_fn = create_reward_function(config)
    validator = SafetyValidator(
        confidence_threshold=config['safety']['confidence_threshold'],
        min_evidence_sentences=config['safety']['min_evidence_sentences'],
        valid_answers=['a', 'b', 'c', 'd']
    )

    print("\nLoading MedQA test set...")
    dataset  = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
    examples = list(dataset)[:args.n]
    print(f"Evaluating on {len(examples)} examples\n")

    strategies = [
        ('always_fast',  'Always-Fast'),
        ('always_deep',  'Always-Deep'),
        ('always_graph', 'Always-Graph'),
        ('random',       'Random'),
    ]

    all_metrics = {}

    for strat_key, strat_name in strategies:
        print(f"\nRunning {strat_name}...")
        res = run_strategy(strat_key, examples, config, kg_arm,
                           reward_fn, validator)
        all_metrics[strat_name] = compute_metrics(res)
        print(f"  {strat_name}: {all_metrics[strat_name]['accuracy']:.1%}")

    print("\nRunning LinUCB Bandit...")
    bandit = LinUCB(
        n_arms=config['bandit']['n_arms'],
        n_features=config['bandit']['n_features'],
        alpha=config['bandit']['alpha'],
    )
    bandit_res = run_strategy('bandit', examples, config, kg_arm,
                              reward_fn, validator, bandit=bandit)
    all_metrics['LinUCB Bandit'] = compute_metrics(bandit_res)
    print(f"  Bandit: {all_metrics['LinUCB Bandit']['accuracy']:.1%}")

    print("\nRunning Thompson Sampling...")
    thompson = ThompsonSampling(n_arms=config['bandit']['n_arms'])
    thompson_res = run_strategy('thompson', examples, config, kg_arm,
                                reward_fn, validator, bandit=thompson)
    all_metrics['Thompson Sampling'] = compute_metrics(thompson_res)
    print(f"  Thompson: {all_metrics['Thompson Sampling']['accuracy']:.1%}")

    print("\n\nMedQA Results Summary")
    print(f"\n{'Strategy':<22} {'Accuracy':>10} {'Mean Reward':>12}")
    print("-" * 46)
    for name, m in all_metrics.items():
        print(f"{name:<22} {m['accuracy']:>9.1%} {m['mean_reward']:>12.4f}")

    # Save
    output_dir = Path("results/medqa")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        'n_examples': len(examples),
        'dataset':    'MedQA-USMLE-4-options',
        'corpus':     'MedRAG textbooks (125,847 chunks)',
        'metrics':    {k: {
            'accuracy':    v['accuracy'],
            'correct':     v['correct'],
            'n':           v['n'],
            'mean_reward': v['mean_reward'],
        } for k, v in all_metrics.items()},
    }
    with open(output_dir / "medqa_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to results/medqa/medqa_results.json")


if __name__ == "__main__":
    main()