from datasets import load_dataset

dataset = load_dataset("MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT")
train = dataset["train"]

malicious = train.filter(lambda row: row["label"] == 1)
print(f"Total malicious rows: {len(malicious)}")

for i in range(5):
    row = malicious[i]
    print(f"\n{'=' * 70}")
    print(f"--- Malicious example {i + 1} ---")
    print(f"context (first 800 chars):\n{str(row['context'])[:800]}")
    print(f"\nuser_intent:\n{row['user_intent']}")
    print(f"\nsource: {row['source']}")