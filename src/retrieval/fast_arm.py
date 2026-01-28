"""
Fast Arm: BM25 keyword-based retrieval
Simple and fast for emergency queries
"""

from rank_bm25 import BM25Okapi


def retrieve_fast(question, context_sentences, top_k=3):
    """
    Use BM25 to get top-k most relevant sentences.
    top_k: how mnay sentrences to return.
    returns list of str, top-k sentences
    """

    # Tokenise 
    tokenized_contexts = [sent.lower().split() for sent in context_sentences]
    tokenized_question = question.lower().split()
    
    # BM25 scoring
    bm25 = BM25Okapi(tokenized_contexts)
    scores = bm25.get_scores(tokenized_question)
    
    # Get top-k
    top_k = min(top_k, len(context_sentences))  
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    return [context_sentences[i] for i in top_indices]


if __name__ == "__main__":
    # Test 
    import json
    
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    
    example = list(data.values())[0]
    question = example['QUESTION']
    contexts = example['CONTEXTS']
    answer = example['final_decision']
    
    print("Testing Fast Arm (BM25)\n")
    print(f"Question: {question}")
    print(f"Gold answer: {answer}")
    print(f"Available contexts: {len(contexts)}")
    print()
    
    retrieved = retrieve_fast(question, contexts, top_k=3)
    
    print(f"Retrieved {len(retrieved)} sentences:")
    for i, sent in enumerate(retrieved, 1):
        print(f"\n{i}. {sent[:150]}...")
    
