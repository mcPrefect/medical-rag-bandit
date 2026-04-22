"""
LLM wrapper for medical question answering.
Uses Qwen2.5-14B-Instruct via transformers pipeline.

Two modes:
  - answer_question()          → yes/no/maybe (for PubMedQA evaluation)
  - answer_question_clinical() → full paragraph (for GP-facing clinical use)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# # Global model (load once, reuse)
# LLM_PIPELINE = None

LLM_MODEL = None
LLM_TOKENIZER = None
ANSWER_TOKEN_IDS = None


def get_llm():
    """Load Qwen2.5-14B-Instruct tokenizer and model (cached)."""
    global LLM_MODEL, LLM_TOKENIZER, ANSWER_TOKEN_IDS
    if LLM_MODEL is None:
        print("Loading LLM model (Qwen2.5-14B-Instruct)...")
        model_name = "Qwen/Qwen2.5-14B-Instruct"
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        LLM_MODEL.eval()
        ANSWER_TOKEN_IDS = _build_answer_token_ids(LLM_TOKENIZER)
        print(ANSWER_TOKEN_IDS)
        print("LLM loaded!")
    return LLM_MODEL, LLM_TOKENIZER


def _build_answer_token_ids(tokenizer):
    """Single-token IDs for yes/no/maybe across casing and spacing variants."""
    categories = {
        "yes":   ["yes", "Yes", "YES", " yes", " Yes"],
        "no":    ["no", "No", "NO", " no", " No"],
        "maybe": ["maybe", "Maybe", "MAYBE", " maybe", " Maybe"],
    }
    result = {}
    for label, forms in categories.items():
        ids = set()
        for form in forms:
            enc = tokenizer.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        result[label] = ids
    return result



def answer_question(question, retrieved_context, max_new_tokens=10):
    """
    Evaluation mode: answer a medical question with yes/no/maybe.
    Used for PubMedQA benchmarking.
    """
    model, tokenizer = get_llm()

    context_text = "\n".join(retrieved_context)

    # system_msg = (
    #     "You are a medical researcher. You will be given context from a "
    #     "biomedical study and a yes/no research question. Based on the "
    #     "findings in the context, answer the question.\n\n"
    #     "Important: Most research questions have a clear yes or no answer "
    #     "based on the study findings. Only answer maybe if the study "
    #     "explicitly reports mixed, inconclusive, or contradictory results. "
    #     "Do not answer maybe simply because you are uncertain.\n\n"
    #     "Respond with a single word: yes, no, or maybe."
    # )

    system_msg = (
    "You are a medical researcher. You will be given context from a "
    "biomedical study and a yes/no research question. Based on the "
    "findings in the context, answer the question.\n\n"
    "- Answer yes if the findings support a positive conclusion.\n"
    "- Answer no if the findings support a negative conclusion.\n"
    "- Answer maybe only if the study explicitly reports conflicting "
    "results that support both yes and no, making it impossible to "
    "determine a clear answer.\n\n"
    "Most questions have a clear yes or no answer. "
    "Respond with a single word: yes, no, or maybe."
)

    user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

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
    generated_ids = output.sequences[0, input_len:]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()

    first_token_probs = torch.softmax(output.scores[0][0].float(), dim=-1)

    def _cat_prob(ids):
        ids = list(ids)
        return first_token_probs[ids].sum().item() if ids else 0.0

    category_probs = {
        "yes":   _cat_prob(ANSWER_TOKEN_IDS["yes"]),
        "no":    _cat_prob(ANSWER_TOKEN_IDS["no"]),
        "maybe": _cat_prob(ANSWER_TOKEN_IDS["maybe"]),
    }
    # print(f"  [debug] yes={category_probs['yes']:.6f} no={category_probs['no']:.6f} maybe={category_probs['maybe']:.6f}")
    first_word = answer_text.split()[0].strip(".,!\"'") if answer_text.split() else ""
    if first_word in ("yes", "no", "maybe"):
        answer = first_word
    elif "yes" in answer_text:
        answer = "yes"
    elif "no" in answer_text:
        answer = "no"
    elif "maybe" in answer_text:
        answer = "maybe"
    else:
        return "maybe", 0.0

    return answer, category_probs[answer]

    # print(f"  [debug] yes={category_probs['yes']:.6f} no={category_probs['no']:.6f} maybe={category_probs['maybe']:.6f}")

    # first_word = answer_text.split()[0].strip(".,!\"'") if answer_text.split() else ""

    # if first_word == "yes":
    #     return "yes"
    # elif first_word == "no":
    #     return "no"
    # elif first_word == "maybe":
    #     return "maybe"
    # elif "yes" in answer_text:
    #     return "yes"
    # elif "no" in answer_text:
    #     return "no"
    # elif "maybe" in answer_text:
    #     return "maybe"
    # else:
    #     return "maybe"


def answer_question_clinical(question, retrieved_context, max_new_tokens=300):
    """
    Clinical mode: answer a medical question with a full explanation.
    Used for GP-facing clinical decision support.

    Returns the same retrieved evidence but as a synthesised, actionable
    clinical response rather than a single word.
    """
    model, tokenizer = get_llm()

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

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict = True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()



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
    predicted, confidence = answer_question(question, contexts)
    print(f"Predicted: {predicted} (confidence {confidence:.6f})")
    print(f"Correct: {predicted == gold_answer}")


    # Clinical mode
    print("\n--- Clinical Mode ---")
    clinical = answer_question_clinical(question, contexts)
    print(clinical)
