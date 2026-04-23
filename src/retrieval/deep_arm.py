"""Deep Arm: Semantic retreival using sentence embeddings"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once and then reuse across class
MODEL = None
def get_model(model_name='all-MiniLM-L6-v2'):
    global MODEL
    if MODEL is None:
        print("Loading sentence transformer model")
        MODEL = SentenceTransformer(model_name, device='cuda')
        print("Model loaded on GPU")

    return MODEL

def retrieve_deep(question, context_sentences, top_k=5, model_name='all-Mini-L6-v2'):
    """Use semantic similarity to get top-k most relevant sentneces"""
    model = get_model(model_name)

    question_embedding = model.encode(question, convert_to_tensor=False)
    context_embeddings = model.encode(context_sentences, convert_to_tensor=False)

    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    context_embeddings = context_embeddings / np.linalg.norm(context_embeddings, axis=1, keepdims=True)

    similarities = np.dot(context_embeddings, question_embedding)

    top_k = min(top_k, len(context_sentences))
    top_indices = np.argsort(similarities)[::-1][:top_k]  # Descending order
    
    return [context_sentences[i] for i in top_indices]

if __name__ == "__main__":
    import json
    import time
    
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    
    example = list(data.values())[0]
    question = example['QUESTION']
    contexts = example['CONTEXTS']
    answer = example['final_decision']
    
    print("Testing Deep Arm\n")
   
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