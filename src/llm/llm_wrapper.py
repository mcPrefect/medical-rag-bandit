"""
Simple LLM wrapper for medical question answering.
Uses Qwen2.5-14B-Instruct via transformers pipeline.
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
            device="cuda",  # Uses GPU
            torch_dtype=torch.float16,  # Use FP16 for speed
        )
        print("LLM loaded!")
    return LLM_PIPELINE


def answer_question(question, retrieved_context, max_new_tokens=10):
    """
    Answer a medical question given retrieved context.
    
    Args:
        question: str, the clinical question
        retrieved_context: list of str, relevant context sentences
        max_new_tokens: int, max length of answer
    
    Returns:
        str: predicted answer ("yes", "no", or "maybe")
    """
    llm = get_llm()

    # Format context
    context_text = "\n".join(retrieved_context)
    
    # system_msg = (
    #     "You are a medical researcher. You will be given context from a "
    #     "biomedical study and a yes/no research question. Based on the "
    #     "findings in the context, answer the question. If the evidence "
    #     "clearly supports a positive conclusion, answer yes. If the "
    #     "evidence clearly supports a negative conclusion, answer no. "
    #     "Only answer maybe if the results are genuinely mixed or "
    #     "inconclusive. Respond with a single word: yes, no, or maybe."
    # )

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
    
#     # Format context
#     context_text = "\n".join([f"- {sent}" for sent in retrieved_context])
    
#     # Create prompt
#     # prompt = f"""You are a medical AI assistant. Answer the question based ONLY on the provided context.
#     prompt = f"""You are a medical expert answering a clinical question. Read the research context carefully and provide a clear answer.

# Context:
# {context_text}

# Question: {question}

# Answer with ONLY one word: "yes", "no", or "maybe".

# Answer:"""
    
#     # Generate
#     messages = [{"role": "user", "content": prompt}]


    response = llm(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Deterministic
        temperature=0.0,
    )
    
    # Extract answer
    answer_text = response[0]["generated_text"][-1]["content"].strip().lower()
    
    # Parse to yes/no/maybe
    # Parse first word to avoid false matches like "beyond" containing "yes"
    first_word = answer_text.split()[0].strip(".,!\"'") if answer_text.split() else ""
    
    if first_word == "yes":
        return "yes"
    elif first_word == "no":
        return "no"
    elif first_word == "maybe":
        return "maybe"
    # Fallback: check anywhere in response
    elif "yes" in answer_text:
        return "yes"
    elif "no" in answer_text:
        return "no"
    elif "maybe" in answer_text:
        return "maybe"
    else:
        return "maybe"


if __name__ == "__main__":
    # Test the LLM
    import json
    
    print("Testing LLM\n")
    
    # Load one example
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
    
    # Get prediction
    print("\nGenerating answer...")
    predicted = answer_question(question, contexts)
    
    print(f"Predicted answer: {predicted}")
    print(f"Correct: {predicted == gold_answer}")
    