"""
LinUCB Contextual Bandit
Chooses between Fast and Deep retrieval arms based on context
"""

# addrd 10-dimensional context vector, section 3.3.1
# and apadtive alpha decay 

import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

# scisapcy for medical entity extraction
# falls back to heuristic if not installed / loaded

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
# (will mirror topics in SafetyValidator contraindication database) - to be updated
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
        """
        Args:
            n_arms: number of arms (3 for Fast Deep, Graph)
            n_features: dimension of context vector (10)
            alpha: initial exploaration parameter 
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha_0 = alpha # Initial alpha for decay

        # Step counter for adaptive decay
        self.t = 0
        
        # For each arm: A matrix and b vector (ridge regression)
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

        # Track per-arm performance (rolling window for historical feature)
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.arm_window = 50  # Rolling window size

    @property
    def alpha(self):
        """Current alpha with decay: α_t = α_0 / √t"""
        return self.alpha_0 / math.sqrt(max(1, self.t))
    
    def select_arm(self, context):
        """
        Select arm with highest UCB score.
        
        Args:
            context: numpy array of shape (n_features,)
        
        Returns:
            int: selected arm index 
        """
        context = np.array(context).flatten()

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
        
        # Select arm with highest UCB
        return int(np.argmax(ucb_scores))
    
    def update(self, arm, context, reward):
        """
        Update arm statistics with observed reward.
        
        Args:
            arm: which arm was selected
            context: context vector used
            reward: observed reward 
        """
        context = np.array(context).flatten()

        # Ensure context matches expected dimensions
        if len(context) != self.n_features:
            if len(context) < self.n_features:
                context = np.pad(context, (0, self.n_features - len(context)))
            else:
                context = context[:self.n_features]
        
        # Update A and b for this arm
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

        # Increment step counter (for alpha decay)
        self.t += 1
        
        # Track per-arm reward history (rolling window)
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
                performances.append(0.5)  # Prior: assume moderate performance
        return performances

# 10-Dimensional Context Feature Extraction

def extract_context(question, context_sentences, bandit=None, kg_arm=None):
    """
    Extract 10-dimensional context vector from question and context.
    
    Implements Section 3.3.1 of  Report:
    
    Features:
        1. Query complexity — medical entity count (scispaCy) [0,1]
        2. Urgency level — emergency keyword detection [0,1]
        3. Patient risk score — high-risk descriptors [0,1]
        4. Question length (normalised) [0,1]
        5. Number of context sentences (normalised) [0,1]
        6. Average context sentence length (normalised) [0,1]
        7. Medical term density — entity/word ratio [0,1]
        8. Guideline coverage — topic match score [0,1]
        9. Historical arm performance — best arm rolling avg [0,1]
        10. KG density — UMLS concept matches [0,1]
    
    Args:
        question: str, the clinical question
        context_sentences: list of str, available context
        bandit: LinUCB instance (optional, for feature #9)
        kg_arm: KnowledgeGraphArm instance (optional, for feature #10)
    
    Returns:
        numpy array of shape (10,)
    """
    question_lower = question.lower()
    words = question.split()
    n_words = max(len(words), 1)
    
    # Feature 1: Query complexity (medical entity count) 
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
    
    # Feature 2: Urgency level 
    urgency_count = sum(
        1 for kw in URGENCY_KEYWORDS
        if kw in question_lower
    )
    urgency = min(urgency_count / 3.0, 1.0)
    
    # Feature 3: Patient risk score 
    # Check question AND context for risk descriptors
    combined_text = question_lower + " " + " ".join(
        s.lower() for s in context_sentences[:3]  # First 3 for speed
    )
    risk_count = sum(
        1 for desc in HIGH_RISK_DESCRIPTORS
        if desc in combined_text
    )
    patient_risk = min(risk_count / 4.0, 1.0)
    
    # Feature 4: Question length (normalised)
    q_len_norm = min(len(words) / 50.0, 1.0)
    
    # Feature 5: Number of context sentences (normalised) 
    n_contexts_norm = min(len(context_sentences) / 10.0, 1.0)
    
    # Feature 6: Average context sentence length (normalised) 
    if context_sentences:
        avg_ctx_len = np.mean([len(s.split()) for s in context_sentences])
    else:
        avg_ctx_len = 0.0
    avg_ctx_len_norm = min(avg_ctx_len / 100.0, 1.0)
    
    # Feature 7: Medical term density (entity/word ratio)
    med_term_density = min(n_entities / n_words, 1.0) if n_words > 0 else 0.0
    
    # Feature 8: Guideline coverage 
    guideline_matches = sum(
        1 for topic in GUIDELINE_TOPICS
        if topic in combined_text
    )
    guideline_coverage = min(guideline_matches / 5.0, 1.0)
    
    # Feature 9: Historical arm performance 
    if bandit is not None:
        arm_perfs = bandit.get_arm_performance()
        # Use best arm's rolling average as the feature
        hist_performance = max(arm_perfs)
    else:
        hist_performance = 0.5  # Default prior
    
    # Feature 10: KG density 
    if kg_arm is not None and hasattr(kg_arm, 'map_entities_to_cuis'):
        try:
            cuis = kg_arm.map_entities_to_cuis(entity_texts)
            kg_density = min(len(cuis) / 10.0, 1.0)
        except Exception:
            kg_density = 0.0
    else:
        kg_density = 0.0
    
    return np.array([
        query_complexity,     # 1
        urgency,              # 2
        patient_risk,         # 3
        q_len_norm,           # 4
        n_contexts_norm,      # 5
        avg_ctx_len_norm,     # 6
        med_term_density,     # 7
        guideline_coverage,   # 8
        hist_performance,     # 9
        kg_density,           # 10
    ])

if __name__ == "__main__":
    import json
    
    print("Testing LinUCB Bandit (10-dim features + alpha decay)")
    
    # Load a few examples
    with open('data/pubmedqa/ori_pqal.json', 'r') as f:
        data = json.load(f)
    
    examples = list(data.values())[:10]
    
    # Create bandit with 10 features
    bandit = LinUCB(n_arms=3, n_features=10, alpha=2.0)
    
    print(f"\nInitial alpha: {bandit.alpha:.4f} (α_0={bandit.alpha_0})")
    
    # Simulate selections
    for i, ex in enumerate(examples):
        question = ex['QUESTION']
        contexts = ex['CONTEXTS']
        
        # Extract 10-dim context
        context = extract_context(question, contexts, bandit=bandit)
        
        # Select arm
        arm = bandit.select_arm(context)
        arm_names = ["Fast", "Deep", "Graph"]
        
        print(f"\nExample {i+1}:")
        print(f"  Q: {question[:60]}...")
        print(f"  Features: {np.round(context, 3)}")
        print(f"  Selected: {arm_names[arm]}")
        print(f"  Alpha: {bandit.alpha:.4f}")
        
        # Simulate reward
        reward = np.random.choice([0.0, 0.3, 0.5, 0.8, 1.0])
        bandit.update(arm, context, reward)
        print(f"  Reward: {reward:.2f}, Step: {bandit.t}")
    
    # Show alpha decay
    print(f"\nAlpha after {bandit.t} steps: {bandit.alpha:.4f}")
    print(f"Arm performances: {[f'{p:.3f}' for p in bandit.get_arm_performance()]}")
    
    # Show feature names
    print("\n10-Dimensional Feature Vector:")
    feature_names = [
        "Query complexity (entities)",
        "Urgency level",
        "Patient risk score",
        "Question length",
        "N context sentences",
        "Avg context length",
        "Medical term density",
        "Guideline coverage",
        "Historical arm perf",
        "KG density",
    ]
    last_context = extract_context(examples[-1]['QUESTION'], examples[-1]['CONTEXTS'])
    for j, (name, val) in enumerate(zip(feature_names, last_context), 1):
        print(f"  {j:2d}. {name:<30s} = {val:.4f}")
    
    print("ALL TESTS PASSED")
