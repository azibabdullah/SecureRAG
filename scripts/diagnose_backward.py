"""
Isolate WHERE the NaN comes from: forward pass, backward pass, or optimizer step.

Runs a handful of manual training steps outside the HF Trainer, checking for
NaN/Inf at each stage separately. This tells us whether the problem is in
gradient computation (likely a ROCm kernel issue with DeBERTa's disentangled
attention) or somewhere else.

Run:
    python scripts/diagnose_backward.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "microsoft/deberta-v3-small"
TRAIN_FILE = Path("data/processed/train.jsonl")
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_STEPS = 10


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"torch: {torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, dtype=torch.float32
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    rows = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= BATCH_SIZE * NUM_STEPS:
                break
            rows.append(json.loads(line))

    print(f"\nRunning {NUM_STEPS} manual training steps...\n")

    for step in range(NUM_STEPS):
        batch = rows[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        texts = [r["text"] for r in batch]
        labels = torch.tensor([r["label"] for r in batch]).to(device)

        inputs = tokenizer(
            texts, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt"
        ).to(device)

        seq_len = inputs["input_ids"].shape[1]

        # --- Forward ---
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits_bad = torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any()
        loss_bad = torch.isnan(loss).any() or torch.isinf(loss).any()

        # --- Backward ---
        optimizer.zero_grad()
        loss.backward()

        grad_nan_params = []
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                grad_nan_params.append(name)

        # --- Optimizer step ---
        optimizer.step()
        weight_nan_params = [
            name for name, p in model.named_parameters()
            if torch.isnan(p).any() or torch.isinf(p).any()
        ]

        print(f"Step {step}: seq_len={seq_len:4d} loss={loss.item():.4f} "
              f"logits_bad={logits_bad} loss_bad={loss_bad} "
              f"grad_nan_count={len(grad_nan_params)} weight_nan_count={len(weight_nan_params)}")

        if grad_nan_params:
            print(f"  >>> FIRST NaN GRADIENTS in {len(grad_nan_params)} params. First 10:")
            for n in grad_nan_params[:10]:
                print(f"      {n}")
            print(f"  >>> This confirms the NaN originates in the BACKWARD pass.")
            print(f"  >>> Batch seq_len was {seq_len}, batch text lengths: "
                  f"{[len(t) for t in texts]}")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()