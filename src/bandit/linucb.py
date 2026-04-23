"""LinUCB Contextual Bandit, chooses between Fast, Deep & Graph retrieval arms based on context"""

import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

# scisapCy for medical entity extraction

_SCISPACY_NLP = None
_SCISPACY_ATTEMPTED = False

def _get_scispacy():
    """Lazy-load scispaCy model (shared singleton)."""
    global _SCISPACY_NLP, _SCISPACY_ATTEMPTED
    if not _SCISPACY_ATTEMPTED:
        _SCISPACY_ATTEMPTED = True
        try:
            import spacy
            _SCISPACY_NLP = spacy.load("en_core_sci_sm")
            logger.info("scispaCy model loaded for context feature extraction")
        except Exception as e:
            logger.warning(f"scispaCy not available ({e}), using heuristic features")
            _SCISPACY_NLP = None
    return _SCISPACY_NLP


# Emergency / urgency keywords
URGENCY_KEYWORDS = {
    "acute", "emergency", "urgent", "stat", "immediately",
    "severe pain", "chest pain", "stroke", "cardiac arrest",
    "anaphylaxis", "haemorrhage", "hemorrhage", "seizure",
    "unconscious", "respiratory failure", "sepsis", "trauma",
    "critical", "life-threatening", "unstable",
}

# High-risk patient descriptors
HIGH_RISK_DESCRIPTORS = {
    "elderly", "geriatric", "pregnant", "pregnancy",
    "infant", "neonatal", "neonate", "pediatric", "paediatric",
    "immunocompromised", "immunosuppressed", "transplant",
    "renal failure", "kidney failure", "liver failure",
    "dialysis", "hiv", "cancer", "terminal", "palliative",
    "multi-morbidity", "comorbidity", "frail",
}

# Guideline / contraindication topic keywords
GUIDELINE_TOPICS = {
    "aspirin", "warfarin", "nsaid", "penicillin", "beta blocker",
    "metformin", "bleeding", "hemophilia", "pregnancy", "kidney",
    "asthma", "allergy", "diabetes", "hypertension", "statin",
    "anticoagulant", "antibiotic", "opioid", "benzodiazepine",
    "contraindication", "interaction", "dosing", "dose",
    "guideline", "protocol", "recommendation",
}


class LinUCB:
    """
    Linear Upper Confidence Bound bandit.
    Learns which arm is best for different contexts.
    """
    
    def __init__(self, n_arms=3, n_features=10, alpha=2.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha_0 = alpha 
        self.t = 0
        
        # For each arm: A matrix and b vector (ridge regression)
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

        # Track per-arm performance
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.arm_window = 50 

    @property
    def alpha(self):
        """Current alpha with decay: α_t = α_0 / √t"""
        return self.alpha_0 / math.sqrt(max(1, self.t))
    
    
    def select_arm_with_probs(self, context):
        """Select arm and return selection probabilities."""
        context = np.array(context).flatten()

        # Ensure context matches expected dimensions
        if len(context) != self.n_features:
            # Pad or truncate if needed (backward compat)
            if len(context) < self.n_features:
                context = np.pad(context, (0, self.n_features - len(context)))
            else:
                context = context[:self.n_features]
        
        current_alpha = self.alpha
        
        ucb_scores = []
        for arm in range(self.n_arms):
            # theta = A^-1 * b (ridge regression solution)
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            
            # UCB = expected reward + exploration bonus
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb = expected_reward + current_alpha * uncertainty
            
            ucb_scores.append(ucb)
        
        ucb_scores = np.array(ucb_scores)
        
        # Convert UCB scores to probabilities via softmax
        # Subtract max for numerical stability
        shifted = ucb_scores - np.max(ucb_scores)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Select arm with highest UCB 
        selected_arm = int(np.argmax(ucb_scores))
        
        return selected_arm, probabilities, ucb_scores
    
    def get_action_probabilities(self, context):
        """Returns arm selection probabilities off-policy evaluation"""
        _, probs, _ = self.select_arm_with_probs(context)
        return probs
    
    def update(self, arm, context, reward):
        """Update arm statistics with observed reward."""
        context = np.array(context).flatten()

        # Ensure context matches expected dimensions
        if len(context) != self.n_features:
            if len(context) < self.n_features:
                context = np.pad(context, (0, self.n_features - len(context)))
            else:
                context = context[:self.n_features]
        
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.t += 1      
        self.arm_rewards[arm].append(reward)
        if len(self.arm_rewards[arm]) > self.arm_window:
            self.arm_rewards[arm] = self.arm_rewards[arm][-self.arm_window:]
    
    def get_arm_performance(self):
        """Get rolling average reward per arm (for context feature #9)."""
        performances = []
        for arm in range(self.n_arms):
            if self.arm_rewards[arm]:
                performances.append(np.mean(self.arm_rewards[arm]))
            else:
                performances.append(0.5)  
        return performances
    
    def save_weights(self, path):
        """Save bandit state to disk """
        import pickle
        state = {
            'n_arms': self.n_arms,
            'n_features': self.n_features,
            'alpha_0': self.alpha_0,
            't': self.t,
            'A': [a.tolist() for a in self.A],
            'b': [b.tolist() for b in self.b],
            'arm_rewards': self.arm_rewards,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Bandit weights saved to {path} (step {self.t})")

    def load_weights(self, path):
        """Load bandit state from disk."""
        import pickle
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.n_arms = state['n_arms']
            self.n_features = state['n_features']
            self.alpha_0 = state['alpha_0']
            self.t = state['t']
            self.A = [np.array(a) for a in state['A']]
            self.b = [np.array(b) for b in state['b']]
            self.arm_rewards = state.get('arm_rewards', [[] for _ in range(self.n_arms)])
            
            logger.info(f"Bandit weights loaded from {path} (step {self.t}, α={self.alpha:.4f})")
            return True
        except FileNotFoundError:
            logger.info(f"No saved weights at {path}, starting fresh")
            return False
        except Exception as e:
            logger.warning(f"Failed to load weights from {path}: {e}")
            return False

# 10-Dimensional Context Feature Extraction
def extract_context(question, context_sentences, bandit=None, kg_arm=None):
    """Extract 10-dimensional context vector from question and context
    for bandit arm selection."""
    question_lower = question.lower()
    words = question.split()
    n_words = max(len(words), 1)
    
    # Query complexity 
    nlp = _get_scispacy()
    if nlp is not None:
        doc = nlp(question)
        n_entities = len(doc.ents)
        entity_texts = [ent.text.lower() for ent in doc.ents]
    else:
        # Fallback: count capitalised multi-char words as proxy
        n_entities = sum(1 for w in words if len(w) > 2 and w[0].isupper())
        entity_texts = [w.lower() for w in words if len(w) > 2 and w[0].isupper()]
    
    query_complexity = min(n_entities / 8.0, 1.0)
    
    # Urgency level 
    urgency_count = sum(
        1 for kw in URGENCY_KEYWORDS
        if kw in question_lower
    )
    urgency = min(urgency_count / 3.0, 1.0)
    
    # Patient risk score 
    combined_text = question_lower + " " + " ".join(
        s.lower() for s in context_sentences[:3]  # First 3 for speed
    )
    risk_count = sum(
        1 for desc in HIGH_RISK_DESCRIPTORS
        if desc in combined_text
    )
    patient_risk = min(risk_count / 4.0, 1.0)
    
    # Question length (normalised)
    q_len_norm = min(len(words) / 50.0, 1.0)
    
    # Number of context sentences (normalised) 
    n_contexts_norm = min(len(context_sentences) / 10.0, 1.0)
    
    # Average context sentence length (normalised) 
    if context_sentences:
        avg_ctx_len = np.mean([len(s.split()) for s in context_sentences])
    else:
        avg_ctx_len = 0.0
    avg_ctx_len_norm = min(avg_ctx_len / 100.0, 1.0)
    
    # Medical term density (entity/word ratio)
    med_term_density = min(n_entities / n_words, 1.0) if n_words > 0 else 0.0
    
    # Guideline coverage 
    guideline_matches = sum(
        1 for topic in GUIDELINE_TOPICS
        if topic in combined_text
    )
    guideline_coverage = min(guideline_matches / 5.0, 1.0)
    
    # Historical arm performance 
    if bandit is not None:
        arm_perfs = bandit.get_arm_performance()
        # Use best arm's rolling average as the feature
        hist_performance = max(arm_perfs)
    else:
        hist_performance = 0.5  # Default prior
    
    # KG density 
    if kg_arm is not None and hasattr(kg_arm, 'map_entities_to_cuis'):
        try:
            cuis = kg_arm.map_entities_to_cuis(entity_texts)
            kg_density = min(len(cuis) / 10.0, 1.0)
        except Exception:
            kg_density = 0.0
    else:
        kg_density = 0.0
    
    return np.array([
        query_complexity,    
        urgency,             
        patient_risk,       
        q_len_norm,          
        n_contexts_norm,     
        avg_ctx_len_norm,   
        med_term_density,     
        guideline_coverage,   
        hist_performance,     
        kg_density,           
    ])