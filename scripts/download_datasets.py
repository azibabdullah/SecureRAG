from datasets import load_dataset

print("Loading Hlyn...")

dataset = load_dataset(
    "hlyn/prompt-injection-judge-deberta-dataset"
)

print(dataset)
print(dataset["train"].features)
print("Rows:", dataset["train"].num_rows)