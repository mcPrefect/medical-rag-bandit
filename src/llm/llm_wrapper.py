"""
LLM wrapper for medical question answering.
Uses Qwen2.5-14B-Instruct via transformers pipeline.

Two modes:
  - answer_question()          → yes/no/maybe (for PubMedQA evaluation)
  - answer_question_clinical() → full paragraph (for GP-facing clinical use)
"""

from transformers import pipeline
import torch


# Global model (load once, reuse)
LLM_PIPELINE = None


def get_llm():
    """Load LLM pipeline (cached)"""
    global LLM_PIPELINE
    if LLM_PIPELINE is None:
        print("Loading LLM model (Qwen2.5-14B-Instruct)...")
        LLM_PIPELINE = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-14B-Instruct",
            device="cuda",
            torch_dtype=torch.float16,
        )
        print("LLM loaded!")
    return LLM_PIPELINE


def answer_question(question, retrieved_context, max_new_tokens=10):
    """
    Evaluation mode: answer a medical question with yes/no/maybe.
    Used for PubMedQA benchmarking.
    """
    llm = get_llm()

    context_text = "\n".join(retrieved_context)

    system_msg = (
        "You are a medical researcher. You will be given context from a "
        "biomedical study and a yes/no research question. Based on the "
        "findings in the context, answer the question.\n\n"
        "Important: Most research questions have a clear yes or no answer "
        "based on the study findings. Only answer maybe if the study "
        "explicitly reports mixed, inconclusive, or contradictory results. "
        "Do not answer maybe simply because you are uncertain.\n\n"
        "Respond with a single word: yes, no, or maybe."
    )

    user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    response = llm(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    answer_text = response[0]["generated_text"][-1]["content"].strip().lower()

    first_word = answer_text.split()[0].strip(".,!\"'") if answer_text.split() else ""

    if first_word == "yes":
        return "yes"
    elif first_word == "no":
        return "no"
    elif first_word == "maybe":
        return "maybe"
    elif "yes" in answer_text:
        return "yes"
    elif "no" in answer_text:
        return "no"
    elif "maybe" in answer_text:
        return "maybe"
    else:
        return "maybe"


def answer_question_clinical(question, retrieved_context, max_new_tokens=300):
    """
    Clinical mode: answer a medical question with a full explanation.
    Used for GP-facing clinical decision support.

    Returns the same retrieved evidence but as a synthesised, actionable
    clinical response rather than a single word.
    """
    llm = get_llm()

    context_text = "\n".join(retrieved_context)

    system_msg = (
        "You are a clinical decision support system for primary care physicians. "
        "You will be given a clinical question and relevant medical evidence "
        "retrieved from the literature.\n\n"
        "Based ONLY on the provided evidence, give a clear, concise clinical "
        "recommendation. Structure your response as:\n"
        "1. A direct answer to the question (1 sentence)\n"
        "2. Key supporting evidence from the context (2-3 sentences)\n"
        "3. Important caveats or safety considerations if any (1 sentence)\n\n"
        "Be specific. Cite numbers, dosages, and outcomes from the evidence "
        "when available. If the evidence is insufficient or contradictory, "
        "say so clearly and recommend specialist referral.\n\n"
        "Do not invent information beyond what the evidence supports. "
        "Do not include disclaimers about being an AI."
    )

    user_msg = f"Evidence:\n{context_text}\n\nClinical question: {question}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    response = llm(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    return response[0]["generated_text"][-1]["content"].strip()


if __name__ == "__main__":
    import json

    print("Testing LLM — both modes\n")

    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)

    example = list(data.values())[0]
    question = example['QUESTION']
    contexts = example['CONTEXTS']
    gold_answer = example['final_decision']

    print(f"Question: {question}\n")
    print(f"Context ({len(contexts)} sentences):")
    for i, ctx in enumerate(contexts, 1):
        print(f"  {i}. {ctx[:100]}...")

    print(f"\nGold answer: {gold_answer}")

    # Evaluation mode
    print("\n--- Evaluation Mode ---")
    predicted = answer_question(question, contexts)
    print(f"Predicted: {predicted}")
    print(f"Correct: {predicted == gold_answer}")

    # Clinical mode
    print("\n--- Clinical Mode ---")
    clinical = answer_question_clinical(question, contexts)
    print(clinical)
