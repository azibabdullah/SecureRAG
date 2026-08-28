from datasets import load_dataset
from collections import Counter
import statistics


def audit_dataset(dataset_id: str, dataset_name: str):
    print("=" * 70)
    print(f"AUDITING DATASET: {dataset_name}")
    print(f"Hugging Face ID: {dataset_id}")
    print("=" * 70)

    print("\n[1] Loading dataset...")

    dataset = load_dataset(dataset_id)

    print("\n[2] Available splits:")
    for split_name, split_data in dataset.items():
        print(f"    {split_name}: {len(split_data):,} examples")

    for split_name, split_data in dataset.items():

        print("\n" + "=" * 70)
        print(f"SPLIT: {split_name}")
        print("=" * 70)

        print("\n[3] Columns:")
        for column in split_data.column_names:
            print(f"    - {column}")

        print("\n[4] Dataset features:")
        print(split_data.features)

        print("\n[5] Number of records:")
        print(f"    {len(split_data):,}")

        # ---------------------------------------------------------
        # Missing values
        # ---------------------------------------------------------

        print("\n[6] Missing values:")

        for column in split_data.column_names:
            missing = 0

            for value in split_data[column]:
                if value is None:
                    missing += 1

            percentage = (
                missing / len(split_data) * 100
                if len(split_data) > 0
                else 0
            )

            print(
                f"    {column}: "
                f"{missing:,} missing "
                f"({percentage:.2f}%)"
            )

        # ---------------------------------------------------------
        # Duplicate rows
        # ---------------------------------------------------------

        print("\n[7] Exact duplicate rows:")

        seen = set()
        duplicate_count = 0

        for row in split_data:
            row_string = str(row)

            if row_string in seen:
                duplicate_count += 1
            else:
                seen.add(row_string)

        print(f"    {duplicate_count:,}")

        # ---------------------------------------------------------
        # Label analysis
        # ---------------------------------------------------------

        print("\n[8] Possible label columns:")

        possible_label_columns = [
            column
            for column in split_data.column_names
            if "label" in column.lower()
            or "class" in column.lower()
            or "target" in column.lower()
        ]

        if not possible_label_columns:
            print("    No obvious label column detected.")

        else:
            for column in possible_label_columns:

                print(f"\n    Label column: {column}")

                values = split_data[column]

                counter = Counter(values)

                for value, count in counter.items():

                    percentage = (
                        count / len(values) * 100
                        if len(values) > 0
                        else 0
                    )

                    print(
                        f"        {repr(value)}: "
                        f"{count:,} "
                        f"({percentage:.2f}%)"
                    )

        # ---------------------------------------------------------
        # Text analysis
        # ---------------------------------------------------------

        print("\n[9] Possible text columns:")

        possible_text_columns = [
            column
            for column in split_data.column_names
            if any(
                keyword in column.lower()
                for keyword in [
                    "text",
                    "prompt",
                    "content",
                    "context",
                    "document",
                    "instruction",
                    "query",
                    "question"
                ]
            )
        ]

        if not possible_text_columns:

            print("    No obvious text column detected.")

        else:

            for column in possible_text_columns:

                print(f"\n    Text column: {column}")

                lengths = []
                empty_count = 0

                for value in split_data[column]:

                    if value is None:
                        continue

                    text = str(value)

                    if not text.strip():
                        empty_count += 1

                    lengths.append(len(text))

                if lengths:

                    print(
                        f"        Empty values: {empty_count:,}"
                    )

                    print(
                        f"        Minimum characters: "
                        f"{min(lengths):,}"
                    )

                    print(
                        f"        Maximum characters: "
                        f"{max(lengths):,}"
                    )

                    print(
                        f"        Average characters: "
                        f"{statistics.mean(lengths):,.2f}"
                    )

                    print(
                        f"        Median characters: "
                        f"{statistics.median(lengths):,.2f}"
                    )

        # ---------------------------------------------------------
        # Sample records
        # ---------------------------------------------------------

        print("\n[10] Sample records:")

        sample_count = min(3, len(split_data))

        for index in range(sample_count):

            print(f"\n--- Example {index + 1} ---")

            row = split_data[index]

            for key, value in row.items():

                value_string = str(value)

                if len(value_string) > 500:
                    value_string = value_string[:500] + "..."

                print(f"{key}: {value_string}")

    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE: {dataset_name}")
    print("=" * 70)


if __name__ == "__main__":

    audit_dataset(
        "hlyn/prompt-injection-judge-deberta-dataset",
        "HLYN Prompt Injection Dataset"
    )