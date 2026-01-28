import json

print("Loading data...")
with open('data/pubmedqa/ori_pqal.json', 'r') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}\n")

# first example
first_id = list(data.keys())[0]
first_example = data[first_id]

print("First example structure:")
print(f"  Fields: {list(first_example.keys())}\n")

print("First example content:")
print(f"  Question: {first_example['QUESTION']}\n")
print(f"  Number of context sentences: {len(first_example['CONTEXTS'])}\n")
print(f"  First context sentence: {first_example['CONTEXTS'][0][:200]}...\n")

# Check what the answer field is called
for key in first_example.keys():
    if 'answer' in key.lower() or 'decision' in key.lower() or 'pred' in key.lower():
        print(f"  Answer field '{key}': {first_example[key]}\n")

# Count context sentences
context_counts = [len(ex['CONTEXTS']) for ex in data.values()]
print(f"Context sentences stats:")
print(f"  Average: {sum(context_counts) / len(context_counts):.1f}")
print(f"  Min: {min(context_counts)}")
print(f"  Max: {max(context_counts)}\n")

# Count answers
answers = []
for ex in data.values():
    for key in ['final_decision', 'reasoning_required_pred', 'ANSWER']:
        if key in ex:
            answers.append(ex[key])
            break

if answers:
    from collections import Counter
    answer_counts = Counter(answers)
    print(f"Answer distribution:")
    for answer, count in answer_counts.items():
        print(f"  {answer}: {count}")