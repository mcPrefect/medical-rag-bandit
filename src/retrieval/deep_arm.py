"""
Deep Arm: Semantic retrival using sentence embeddings
Slower but undersatnds meaning beyond keywords
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once and then reuse across calss
MODEL = None
def get_model():
    global MODEL
    if MODEL is None:
        print("Loading sentence transformer model (cached)")
        MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cuda') # Fast & Lightweight for MVP
        print("Model loaded on GPU")

    return MODEL

def retrieve_deep(question, context_sentences, top_k=5):
    """
    Use semantic similarity to get top-k most relevant sentneces.
    
    :param question: str
    :param context_sentences: list of str
    :param top_k: how many sentences to return
    
    Returns list of str: top-k sentences
    """
    model = get_model()

    # Encode question and contexts 
    question_embedding = model.encode(question, convert_to_tensor=False)
    context_embeddings = model.encode(context_sentences, convert_to_tensor=False)

    # Compute cosine similarities & Normlaise embeddings
    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    context_embeddings = context_embeddings / np.linalg.norm(context_embeddings, axis=1, keepdims=True)

    # Cosine similarity = dot product of normalised vectors
    similarities = np.dot(context_embeddings, question_embedding)

    # Get top-k
    top_k = min(top_k, len(context_sentences))
    top_indices = np.argsort(similarities)[::-1][:top_k]  # Descending order
    
    return [context_sentences[i] for i in top_indices]

if __name__ == "__main__":
    # Test on one example
    import json
    import time
    
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    
    example = list(data.values())[0]
    question = example['QUESTION']
    contexts = example['CONTEXTS']
    answer = example['final_decision']
    
    print("Testing Deep Arm (Semantic)\n")
   
    print(f"Question: {question}")
    print(f"Gold answer: {answer}")
    print(f"Available contexts: {len(contexts)}")
    print()
    
    start = time.time()
    retrieved = retrieve_deep(question, contexts, top_k=5)
    elapsed = time.time() - start
    
    print(f"Retrieved {len(retrieved)} sentences in {elapsed:.2f}s:")
    for i, sent in enumerate(retrieved, 1):
        print(f"\n{i}. {sent[:150]}...")