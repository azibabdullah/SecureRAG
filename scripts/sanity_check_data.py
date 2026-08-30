"""
Scan data/processed/train.jsonl for rows that could cause numerical instability
during training: literal "nan" strings (a classic pandas gotcha - NaN.astype(str)
produces the string "nan", which can slip past a min-length filter), empty/whitespace-
only text, non-string text, or labels outside {0,1}.

Run:
    python scripts/sanity_check_data.py
"""

import json
from pathlib import Path
from collections import Counter

PATHS = [
    Path("data/processed/train.jsonl"),
    Path("data/processed/validation.jsonl"),
    Path("data/processed/test.jsonl"),
]

SUSPICIOUS_STRINGS = {"nan", "none", "null", "n/a", "na", ""}


def check_file(path: Path):
    print(f"\n=== {path} ===")
    if not path.exists():
        print("  (file not found, skipping)")
        return

    total = 0
    issues = Counter()
    example_issues = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues["invalid_json"] += 1
                continue

            text = row.get("text")
            label = row.get("label")

            if not isinstance(text, str):
                issues["text_not_string"] += 1
                if len(example_issues) < 5:
                    example_issues.append((line_num, "text_not_string", row))
                continue

            if text.strip().lower() in SUSPICIOUS_STRINGS:
                issues["suspicious_literal_string"] += 1
                if len(example_issues) < 5:
                    example_issues.append((line_num, "suspicious_literal_string", row))

            if len(text.strip()) < 3:
                issues["too_short"] += 1

            if label not in (0, 1):
                issues["bad_label"] += 1
                if len(example_issues) < 5:
                    example_issues.append((line_num, "bad_label", row))

    print(f"  Total rows: {total}")
    if issues:
        print(f"  Issues found: {dict(issues)}")
        print("  Example problem rows:")
        for line_num, issue_type, row in example_issues:
            print(f"    line {line_num} [{issue_type}]: {row}")
    else:
        print("  No issues found - clean.")


if __name__ == "__main__":
    for p in PATHS:
        check_file(p)