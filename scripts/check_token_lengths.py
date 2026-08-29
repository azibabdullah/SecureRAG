from datasets import load_dataset
from transformers import AutoTokenizer
import statistics


def check_token_lengths(dataset_id: str, dataset_name: str, text_column: str = "text", sample_size: int = 5000):
    print("=" * 70)
    print(f"TOKEN LENGTH CHECK: {dataset_name}")
    print("=" * 70)

    print("\nLoading dataset (should be fast if already cached)...")
    dataset = load_dataset(dataset_id)

    print("Loading DeBERTa-v3-small tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

    for split_name, split_data in dataset.items():
        n = min(sample_size, len(split_data))
        print(f"\nSplit '{split_name}': sampling {n:,} of {len(split_data):,} rows...")

        # Simple even-spaced sample rather than pandas .sample() to avoid an extra dependency here
        step = max(1, len(split_data) // n)
        indices = list(range(0, len(split_data), step))[:n]

        lengths = []
        for i in indices:
            text = str(split_data[i][text_column])
            tokens = tokenizer.encode(text, truncation=False)
            lengths.append(len(tokens))

        print(f"\nToken length stats (n={len(lengths):,}):")
        print(f"    min: {min(lengths)}")
        print(f"    max: {max(lengths)}")
        print(f"    mean: {statistics.mean(lengths):.1f}")
        print(f"    median: {statistics.median(lengths)}")

        buckets = {"<128": 0, "128-256": 0, "256-512": 0, ">512": 0}
        for l in lengths:
            if l < 128:
                buckets["<128"] += 1
            elif l < 256:
                buckets["128-256"] += 1
            elif l <= 512:
                buckets["256-512"] += 1
            else:
                buckets[">512"] += 1

        print("\nToken length buckets:")
        for bucket, count in buckets.items():
            pct = count / len(lengths) * 100
            print(f"    {bucket}: {count:,} ({pct:.2f}%)")

        over_512_pct = buckets[">512"] / len(lengths) * 100
        print(f"\n% that would be TRUNCATED at max_length=512: {over_512_pct:.2f}%")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    check_token_lengths(
        "hlyn/prompt-injection-judge-deberta-dataset",
        "HLYN Prompt Injection Dataset"
    )