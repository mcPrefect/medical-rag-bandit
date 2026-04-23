"""
Safety Validator: Multi-layer validation before delivering answers
Implements self-protection pillar of autonomous system

4 Independent Checks (all must pass):
1. Confidence Threshold - Is LLM confident enough?
2. Evidence Sufficiency - Did we retrieve enough context?
3. Contraindication Check - Any dangerous recommendations?
4. Sanity Check - Does answer make sense?
"""

import re
from typing import List, Tuple, Dict


class SafetyValidator:
    """
    Multi-layer safety validator for medical AI responses.
    Each layer can independently trigger abstention.
    """
    
    def __init__(
        self,
        confidence_threshold=0.7,
        min_evidence_sentences=2,
        enable_contraindication_check=True,
        enable_sanity_check=True,
        valid_answers=None
    ):
        self.confidence_threshold = confidence_threshold
        self.min_evidence_sentences = min_evidence_sentences
        self.enable_contraindication_check = enable_contraindication_check
        self.enable_sanity_check = enable_sanity_check
        self.valid_answers = valid_answers or ['yes', 'no', 'maybe']
        

        self.contraindications = {
            # Anticoagulants / bleeding risk
            ('aspirin', 'bleeding'),
            ('aspirin', 'hemophilia'),
            ('aspirin', 'bleeding disorder'),
            ('warfarin', 'pregnancy'),
            ('warfarin', 'nsaid'),
            ('warfarin', 'aspirin'),
            ('heparin', 'bleeding disorder'),
            ('heparin', 'thrombocytopenia'),
            ('clopidogrel', 'bleeding'),
            ('thrombolytic', 'hemorrhage'),
            ('thrombolytic', 'recent surgery'),

            # NSAIDs / renal / GI
            ('nsaid', 'kidney disease'),
            ('nsaid', 'renal failure'),
            ('nsaid', 'peptic ulcer'),
            ('nsaid', 'heart failure'),
            ('nsaid', 'pregnancy'),
            ('ibuprofen', 'kidney disease'),
            ('ibuprofen', 'peptic ulcer'),
            ('naproxen', 'renal failure'),

            # Antibiotics / allergy
            ('penicillin', 'penicillin allergy'),
            ('amoxicillin', 'penicillin allergy'),
            ('cephalosporin', 'penicillin allergy'),
            ('sulfonamide', 'sulfa allergy'),
            ('tetracycline', 'pregnancy'),
            ('fluoroquinolone', 'tendon disorder'),
            ('fluoroquinolone', 'myasthenia gravis'),

            # Cardiac / respiratory
            ('beta blocker', 'asthma'),
            ('beta blocker', 'copd'),
            ('beta blocker', 'bradycardia'),
            ('calcium channel blocker', 'heart failure'),
            ('digoxin', 'hypokalaemia'),
            ('ace inhibitor', 'pregnancy'),
            ('ace inhibitor', 'hyperkalaemia'),
            ('nitrate', 'hypotension'),

            # Diabetes / metabolic
            ('metformin', 'kidney failure'),
            ('metformin', 'renal impairment'),
            ('insulin', 'hypoglycaemia'),
            ('thiazolidinedione', 'heart failure'),

            # CNS / psychiatric
            ('maoi', 'ssri'),
            ('maoi', 'tyramine'),
            ('ssri', 'maoi'),
            ('opioid', 'respiratory depression'),
            ('benzodiazepine', 'respiratory depression'),
            ('lithium', 'nsaid'),
            ('lithium', 'dehydration'),
        }
    
    def validate(
        self,
        question: str,
        retrieved_context: List[str],
        predicted_answer: str,
        confidence: float = None
    ) -> Tuple[bool, str, Dict]:
        """Validate a response through all safety layers."""
        details = {}
        
        # Confidence Threshold
        confidence_pass, confidence_reason = self._check_confidence(confidence)
        details['confidence'] = {
            'pass': confidence_pass,
            'score': confidence,
            'threshold': self.confidence_threshold,
            'reason': confidence_reason
        }
        
        if not confidence_pass:
            return False, confidence_reason, details
        
        # Evidence Sufficiency
        evidence_pass, evidence_reason = self._check_evidence_sufficiency(retrieved_context)
        details['evidence'] = {
            'pass': evidence_pass,
            'num_sentences': len(retrieved_context),
            'min_required': self.min_evidence_sentences,
            'reason': evidence_reason
        }
        
        if not evidence_pass:
            return False, evidence_reason, details
        
        # Contraindication Check
        if self.enable_contraindication_check:
            contra_pass, contra_reason = self._check_contraindications(
                question, retrieved_context, predicted_answer
            )
            details['contraindications'] = {
                'pass': contra_pass,
                'reason': contra_reason
            }
            
            if not contra_pass:
                return False, contra_reason, details
        
        # Sanity Check
        if self.enable_sanity_check:
            sanity_pass, sanity_reason = self._check_sanity(
                question, predicted_answer
            )
            details['sanity'] = {
                'pass': sanity_pass,
                'reason': sanity_reason
            }
            
            if not sanity_pass:
                return False, sanity_reason, details
        
        return True, "All safety checks passed", details
    

    def _check_confidence(self, confidence: float) -> Tuple[bool, str]:
        """Check if confidence meets threshold, if missing its treated as an abstain"""
        if confidence is None:
            return False, "No confidence score provided"
        if confidence < self.confidence_threshold:
            return False, f"Low confidence ({confidence:.2f} < {self.confidence_threshold})"
        return True, f"Confidence acceptable ({confidence:.2f})"
    
    def _check_evidence_sufficiency(self, retrieved_context: List[str]) -> Tuple[bool, str]:
        """Check if enough evidence was retrieved."""
        num_sentences = len(retrieved_context)
        
        if num_sentences < self.min_evidence_sentences:
            return False, f"Insufficient evidence ({num_sentences} < {self.min_evidence_sentences} sentences)"
        
        # Check if context is too short
        avg_length = sum(len(s.split()) for s in retrieved_context) / len(retrieved_context)
        if avg_length < 5:  # Very short sentences
            return False, f"Evidence too brief (avg {avg_length:.1f} words/sentence)"
        
        return True, f"Evidence sufficient ({num_sentences} sentences)"
    
    def _check_contraindications(
        self,
        question: str,
        retrieved_context: List[str],
        predicted_answer: str
    ) -> Tuple[bool, str]:
        """Check for dangerous drug-condition interactions."""
        # Simple keyword-based check 
        question_lower = question.lower()
        
        # Check each known contraindication
        for drug, condition in self.contraindications:
            if condition in question_lower and drug in question_lower:
                return False, f"Potential contraindication: {drug} with {condition}"
        
        return True, "No contraindications detected"
    
    def _check_sanity(self, question: str, predicted_answer: str) -> Tuple[bool, str]:
        """Basic sanity checks on the answer."""
        # Check 1: Answer should be yes/no/maybe
        # valid_answers = ['yes', 'no', 'maybe']
        if predicted_answer.lower() not in self.valid_answers:
            return False, f"Invalid answer format: '{predicted_answer}'"
        
        # Check 2: Question asks yes/no question (has '?')
        if '?' not in question:
            return False, "Question format unclear (no question mark)"
        
        return True, "Answer format valid"


if __name__ == "__main__":
    print("Testing Safety Validator")
    
    validator = SafetyValidator(
        confidence_threshold=0.7,
        min_evidence_sentences=2
    )
    
    # Test Case 1: Should pass (good answer)
    print("\nTest 1: Normal case (should pass)")
    safe, reason, details = validator.validate(
        question="Does aspirin reduce cardiovascular risk?",
        retrieved_context=[
            "Studies show aspirin reduces heart attack risk in high-risk patients.",
            "Daily low-dose aspirin is recommended for cardiovascular prevention."
        ],
        predicted_answer="yes",
        confidence=0.85
    )
    print(f"  Safe: {safe}")
    print(f"  Reason: {reason}")
    
    # Test Case 2: Should fail (low confidence)
    print("\nTest 2: Low confidence (should abstain)")
    safe, reason, details = validator.validate(
        question="Is this drug effective?",
        retrieved_context=["Some evidence exists.", "Results are mixed."],
        predicted_answer="maybe",
        confidence=0.5  # Below threshold
    )
    print(f"  Safe: {safe}")
    print(f"  Reason: {reason}")
    
    # Test Case 3: Should fail (insufficient evidence)
    print("\nTest 3: Insufficient evidence (should abstain)")
    safe, reason, details = validator.validate(
        question="What causes headaches?",
        retrieved_context=["Many factors."],  # Only 1 sentence
        predicted_answer="maybe",
        confidence=0.9
    )
    print(f"  Safe: {safe}")
    print(f"  Reason: {reason}")
    
    # Test Case 4: Should fail (contraindication)
    print("\nTest 4: Contraindication detected (should abstain)")
    safe, reason, details = validator.validate(
        question="Patient has bleeding disorder. What medication?",
        retrieved_context=[
            "Aspirin is commonly used for pain relief.",
            "Patient history shows bleeding problems."
        ],
        predicted_answer="yes",
        confidence=0.9
    )
    print(f"  Safe: {safe}")
    print(f"  Reason: {reason}")
