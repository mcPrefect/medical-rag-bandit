"""
Simple LLM wrapper for medical question answering.
Uses Qwen2.5-3B-Instruct via transformers pipeline.
"""

from transformers import pipeline
import torch


# Global model (load once, reuse)
LLM_PIPELINE = None


def get_llm():
    """Load LLM pipeline (cached)"""
    global LLM_PIPELINE
    if LLM_PIPELINE is None:
        print("Loading LLM model (Qwen2.5-3B-Instruct)...")
        LLM_PIPELINE = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct",
            device="cuda",  # Uses GPU
            torch_dtype=torch.float16,  # Use FP16 for speed
        )
        print("LLM loaded!")
    return LLM_PIPELINE


def answer_question(question, retrieved_context, max_new_tokens=50):
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
    context_text = "\n".join([f"- {sent}" for sent in retrieved_context])
    
    # Create prompt
    prompt = f"""You are a medical AI assistant. Answer the question based ONLY on the provided context.

Context:
{context_text}

Question: {question}

Answer with ONLY one word: "yes", "no", or "maybe".

Answer:"""
    
    # Generate
    messages = [{"role": "user", "content": prompt}]
    response = llm(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Deterministic
        temperature=0.0,
    )
    
    # Extract answer
    answer_text = response[0]["generated_text"][-1]["content"].strip().lower()
    
    # Parse to yes/no/maybe
    if "yes" in answer_text:
        return "yes"
    elif "no" in answer_text:
        return "no"
    elif "maybe" in answer_text:
        return "maybe"
    else:
        # Default to "maybe" if unclear
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
    