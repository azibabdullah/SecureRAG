from datasets import load_dataset
from collections import Counter
import statistics
from pathlib import Path
import sys


# ============================================================
# Configuration
# ============================================================

DOCS_DIR = Path("docs")


# ============================================================
# Audit function
# ============================================================

def audit_dataset(dataset_id: str, dataset_name: str, output_file: str):

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DOCS_DIR / output_file

    # Capture all output so we can print it AND save it.
    lines = []

    def log(message=""):
        print(message)
        lines.append(str(message))

    log("=" * 70)
    log(f"AUDITING DATASET: {dataset_name}")
    log(f"Hugging Face ID: {dataset_id}")
    log("=" * 70)

    log("\n[1] Loading dataset...")

    try:
        dataset = load_dataset(dataset_id)
    except Exception as error:
        log(f"\nERROR: Could not load dataset.")
        log(str(error))
        return

    log("\n[2] Available splits:")

    for split_name, split_data in dataset.items():
        log(f"    {split_name}: {len(split_data):,} examples")

    for split_name, split_data in dataset.items():

        log("\n" + "=" * 70)
        log(f"SPLIT: {split_name}")
        log("=" * 70)

        # --------------------------------------------------------
        # Columns
        # --------------------------------------------------------

        log("\n[3] Columns:")

        for column in split_data.column_names:
            log(f"    - {column}")

        # --------------------------------------------------------
        # Features
        # --------------------------------------------------------

        log("\n[4] Dataset features:")
        log(str(split_data.features))

        # --------------------------------------------------------
        # Number of records
        # --------------------------------------------------------

        log("\n[5] Number of records:")
        log(f"    {len(split_data):,}")

        # --------------------------------------------------------
        # Missing values
        # --------------------------------------------------------

        log("\n[6] Missing values:")

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

            log(
                f"    {column}: "
                f"{missing:,} missing "
                f"({percentage:.2f}%)"
            )

        # --------------------------------------------------------
        # Duplicate rows
        # --------------------------------------------------------

        log("\n[7] Exact duplicate rows:")

        seen = set()
        duplicate_count = 0

        for row in split_data:

            row_string = str(row)

            if row_string in seen:
                duplicate_count += 1
            else:
                seen.add(row_string)

        log(f"    {duplicate_count:,}")

        # --------------------------------------------------------
        # Label analysis
        # --------------------------------------------------------

        log("\n[8] Possible label columns:")

        possible_label_columns = [
            column
            for column in split_data.column_names
            if (
                "label" in column.lower()
                or "class" in column.lower()
                or "target" in column.lower()
            )
        ]

        if not possible_label_columns:

            log("    No obvious label column detected.")

        else:

            for column in possible_label_columns:

                log(f"\n    Label column: {column}")

                values = split_data[column]

                counter = Counter(values)

                for value, count in counter.items():

                    percentage = (
                        count / len(values) * 100
                        if len(values) > 0
                        else 0
                    )

                    log(
                        f"        {repr(value)}: "
                        f"{count:,} "
                        f"({percentage:.2f}%)"
                    )

        # --------------------------------------------------------
        # Text analysis
        # --------------------------------------------------------

        log("\n[9] Possible text columns:")

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

            log("    No obvious text column detected.")

        else:

            for column in possible_text_columns:

                log(f"\n    Text column: {column}")

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

                    log(
                        f"        Empty values: "
                        f"{empty_count:,}"
                    )

                    log(
                        f"        Minimum characters: "
                        f"{min(lengths):,}"
                    )

                    log(
                        f"        Maximum characters: "
                        f"{max(lengths):,}"
                    )

                    log(
                        f"        Average characters: "
                        f"{statistics.mean(lengths):,.2f}"
                    )

                    log(
                        f"        Median characters: "
                        f"{statistics.median(lengths):,.2f}"
                    )

        # --------------------------------------------------------
        # Sample records
        # --------------------------------------------------------

        log("\n[10] Sample records:")

        sample_count = min(3, len(split_data))

        for index in range(sample_count):

            log(f"\n--- Example {index + 1} ---")

            row = split_data[index]

            for key, value in row.items():

                value_string = str(value)

                if len(value_string) > 500:
                    value_string = value_string[:500] + "..."

                log(f"{key}: {value_string}")

    log("\n" + "=" * 70)
    log(f"AUDIT COMPLETE: {dataset_name}")
    log("=" * 70)

    # ============================================================
    # Save audit
    # ============================================================

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(f"\nAudit saved to:")
    print(output_path.resolve())


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 4:

        print(
            "Usage:\n"
            "python scripts\\audit_dataset.py "
            "<dataset_id> <dataset_name> <output_file>"
        )

        sys.exit(1)

    dataset_id = sys.argv[1]
    dataset_name = sys.argv[2]
    output_file = sys.argv[3]

    audit_dataset(
        dataset_id,
        dataset_name,
        output_file
    )