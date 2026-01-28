"""
LinUCB Contextual Bandit
Chooses between Fast and Deep retrieval arms based on context
"""

import numpy as np


class LinUCB:
    """
    Linear Upper Confidence Bound bandit.
    Learns which arm is best for different contexts.
    """
    
    def __init__(self, n_arms=2, n_features=4, alpha=1.0):
        """
        Args:
            n_arms: number of arms (2 for Fast vs Deep)
            n_features: dimension of context vector
            alpha: exploration parameter (higher = more exploration)
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # For each arm: A matrix and b vector (ridge regression)
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
    
    def select_arm(self, context):
        """
        Select arm with highest UCB score.
        
        Args:
            context: numpy array of shape (n_features,)
        
        Returns:
            int: selected arm index (0 or 1)
        """
        context = np.array(context).flatten()
        
        ucb_scores = []
        for arm in range(self.n_arms):
            # theta = A^-1 * b (ridge regression solution)
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            
            # UCB = expected reward + exploration bonus
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb = expected_reward + self.alpha * uncertainty
            
            ucb_scores.append(ucb)
        
        # Select arm with highest UCB
        return int(np.argmax(ucb_scores))
    
    def update(self, arm, context, reward):
        """
        Update arm statistics with observed reward.
        
        Args:
            arm: which arm was selected
            context: context vector used
            reward: observed reward (0 or 1 for correct/incorrect)
        """
        context = np.array(context).flatten()
        
        # Update A and b for this arm
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


def extract_context(question, context_sentences):
    """
    Extract context features from question and available context.
    
    Features:
        1. Question length (normalized)
        2. Number of context sentences (normalized)
        3. Average context sentence length (normalized)
        4. Complexity proxy: unique medical terms (normalized)
    
    Returns:
        numpy array of shape (4,)
    """
    # Question length
    q_len = len(question.split())
    q_len_norm = min(q_len / 50.0, 1.0)  # Normalise, cap at 50 words
    
    # Number of contexts
    n_contexts = len(context_sentences)
    n_contexts_norm = min(n_contexts / 10.0, 1.0)  # Cap at 10
    
    # Average context length
    avg_ctx_len = np.mean([len(s.split()) for s in context_sentences])
    avg_ctx_len_norm = min(avg_ctx_len / 100.0, 1.0)  # Cap at 100
    
    # Complexity: count capitalised words (proxy for medical terms)
    words = question.split()
    capital_words = sum(1 for w in words if w[0].isupper() and len(w) > 2)
    complexity_norm = min(capital_words / 5.0, 1.0)  # Cap at 5
    
    return np.array([q_len_norm, n_contexts_norm, avg_ctx_len_norm, complexity_norm])


if __name__ == "__main__":
    # Test the bandit
    import json
    
    print("Testing LinUCB Bandit\n")
    
    # Load a few examples
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    
    examples = list(data.values())[:5]
    
    # Create bandit
    bandit = LinUCB(n_arms=2, n_features=4, alpha=1.0)
    
    # Simulate selections
    for i, ex in enumerate(examples):
        question = ex['QUESTION']
        contexts = ex['CONTEXTS']
        
        # Extract context
        context = extract_context(question, contexts)
        
        # Select arm
        arm = bandit.select_arm(context)
        arm_name = "Fast" if arm == 0 else "Deep"
        
        print(f"\nExample {i+1}:")
        print(f"  Question: {question[:60]}...")
        print(f"  Context features: {context}")
        print(f"  Selected arm: {arm_name}")
        
        # Simulate reward (random for now)
        reward = np.random.choice([0, 1])
        bandit.update(arm, context, reward)
        print(f"  Reward: {reward}")
