"""
Safety Validator Edge Case Testing
Tests validator on 20 crafted cases that should/shouldn't abstain
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.safety.validator import SafetyValidator


def run_edge_case_tests(test_file="tests/safety_edge_cases.json"):
    """Run all edge case tests and report results."""
    
    print("SAFETY VALIDATOR EDGE CASE TESTING")
    
    # Load test cases
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    test_cases = data['test_cases']
    print(f"\nLoaded {len(test_cases)} test cases\n")
    
    # Initialize validator
    validator = SafetyValidator(
        confidence_threshold=0.7,
        min_evidence_sentences=2
    )
    
    # Track results
    results = {
        'total': len(test_cases),
        'passed': 0,
        'failed': 0,
        'false_positives': 0,  # Should pass but abstained
        'false_negatives': 0,  # Should abstain but passed
        'failures': []
    }
    
    # Run each test
    for test in test_cases:
        test_id = test['id']
        category = test['category']
        question = test['question']
        context = test['retrieved_context']
        answer = test['predicted_answer']
        should_abstain = test['should_abstain']
        
        # Run validator
        is_safe, reason, details = validator.validate(
            question=question,
            retrieved_context=context,
            predicted_answer=answer,
            confidence=None
        )
        
        did_abstain = not is_safe
        
        # Check if test passed
        test_passed = (did_abstain == should_abstain)
        
        if test_passed:
            results['passed'] += 1
            status = "✓ PASS"
        else:
            results['failed'] += 1
            status = "x FAIL"
            
            # Categorize failure type
            if should_abstain and not did_abstain:
                results['false_negatives'] += 1
                failure_type = "FALSE NEGATIVE (should abstain but didn't)"
            else:
                results['false_positives'] += 1
                failure_type = "FALSE POSITIVE (abstained unnecessarily)"
            
            results['failures'].append({
                'id': test_id,
                'category': category,
                'type': failure_type,
                'question': question[:60],
                'reason': reason
            })
        
        # Print result
        print(f"Test {test_id:2d} [{category:30s}] {status}")
        if not test_passed:
            print(f"         Expected: {'Abstain' if should_abstain else 'Pass'}, "
                  f"Got: {'Abstain' if did_abstain else 'Pass'}")
            print(f"         Reason: {reason}")
        
    # Print summary
    print("TEST SUMMARY")
    
    pass_rate = (results['passed'] / results['total']) * 100
    
    print(f"\nTotal Tests: {results['total']}")
    print(f"Passed: {results['passed']} ({pass_rate:.1f}%)")
    print(f"Failed: {results['failed']}")
    
    if results['failed'] > 0:
        print(f"\nFailure Breakdown:")
        print(f"  False Negatives: {results['false_negatives']} (missed dangerous cases)")
        print(f"  False Positives: {results['false_positives']} (rejected safe cases)")
        
        print(f"\nFailed Tests:")
        for failure in results['failures']:
            print(f"  #{failure['id']} [{failure['category']}]")
            print(f"      {failure['type']}")
            print(f"      Q: {failure['question']}...")
            print(f"      Reason: {failure['reason']}")
    
    
    return results


if __name__ == "__main__":
    results = run_edge_case_tests()
