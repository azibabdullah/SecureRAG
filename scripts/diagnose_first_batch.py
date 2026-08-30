"""
Diagnose which training example(s) cause NaN/Inf during a forward pass.

Loads a FRESH (untrained) DeBERTa model - same as train_deberta.py would at
step 0 - and runs every batch of train.jsonl through it in inference mode
(no backward pass, no optimizer). Any batch whose output logits contain
NaN or Inf gets flagged, and we narrow down to the individual example.

Run:
    python scripts/diagnose_first_batch.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "microsoft/deberta-v3-small"
TRAIN_FILE = Path("data/processed/train.jsonl")
MAX_LENGTH = 512
BATCH_SIZE = 16
MAX_ROWS_TO_SCAN = 50000  # cap for time - increase if the bad row isn't in the first 50K


def load_rows(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading tokenizer and FRESH (untrained) model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, dtype=torch.float32
    )
    model.to(device)
    model.eval()

    print(f"Loading first {MAX_ROWS_TO_SCAN} rows of {TRAIN_FILE}...")
    rows = load_rows(TRAIN_FILE, MAX_ROWS_TO_SCAN)
    print(f"Loaded {len(rows)} rows. Scanning in batches of {BATCH_SIZE}...")

    bad_found = False
    with torch.no_grad():
        for batch_start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[batch_start:batch_start + BATCH_SIZE]
            texts = [r["text"] for r in batch_rows]

            inputs = tokenizer(
                texts, truncation=True, max_length=MAX_LENGTH,
                padding=True, return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                bad_found = True
                print(f"\n*** BAD BATCH at rows {batch_start}-{batch_start + len(batch_rows)} ***")
                # Narrow down to the individual example(s) within this batch
                for i, row in enumerate(batch_rows):
                    single_input = tokenizer(
                        row["text"], truncation=True, max_length=MAX_LENGTH,
                        return_tensors="pt"
                    ).to(device)
                    single_out = model(**single_input)
                    if torch.isnan(single_out.logits).any() or torch.isinf(single_out.logits).any():
                        token_count = single_input["input_ids"].shape[1]
                        print(f"  BAD ROW at global index {batch_start + i}:")
                        print(f"    source: {row.get('source')}")
                        print(f"    label: {row.get('label')}")
                        print(f"    token count (after truncation): {token_count}")
                        print(f"    char length: {len(row['text'])}")
                        print(f"    text (first 500 chars): {row['text'][:500]!r}")
                        print(f"    text (last 200 chars): {row['text'][-200:]!r}")

            if batch_start % 3200 == 0:
                print(f"  ...scanned {batch_start}/{len(rows)} rows, bad_found={bad_found}")

    if not bad_found:
        print(f"\nNo NaN/Inf found in the first {len(rows)} rows. "
              f"Increase MAX_ROWS_TO_SCAN and re-run, or the trigger may need "
              f"an actual optimizer step (not just forward pass) to manifest.")
    else:
        print("\nScan complete. Review the BAD ROW(s) above.")


if __name__ == "__main__":
    main()