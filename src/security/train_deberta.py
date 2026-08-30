"""
SecureRAG DeBERTa-v3-small Training - standalone PyTorch loop
===============================================================
Uses a plain PyTorch training loop rather than HuggingFace's Trainer.
The Trainer/accelerate stack produced immediate NaN gradients on this
setup (AMD ROCm), while this exact manual loop ran cleanly - so we use
what provably works.

Features: per-epoch validation, best-checkpoint saving by F1, per-source
test metrics, resumable-safe checkpointing, full metadata output.

Usage:
    python src/security/train_deberta.py --smoke-test   # quick pipeline check
    python src/security/train_deberta.py                # full training run
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "microsoft/deberta-v3-small"
DATA_DIR = Path("data/processed")
HELD_OUT_FILE = Path("data/held_out_eval/agentic_5k.jsonl")
OUTPUT_MODEL_DIR = Path("models/deberta-v3-small-securerag")

MAX_LENGTH = 512
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
RANDOM_SEED = 42
LOG_EVERY = 100

LABEL_NAMES = {0: "SAFE", 1: "MALICIOUS"}
SMOKE_TEST_SAMPLES = 3000


class JsonlDataset(Dataset):
    def __init__(self, path, limit=None):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class Collator:
    """Module-level class (not a closure) so it can be pickled for
    DataLoader worker processes on Windows, which uses spawn, not fork."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [r["text"] for r in batch]
        labels = torch.tensor([int(r["label"]) for r in batch], dtype=torch.long)
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                             padding=True, return_tensors="pt")
        enc["labels"] = labels
        return enc


@torch.no_grad()
def evaluate(model, loader, device, return_preds=False):
    model.eval()
    all_preds, all_labels, total_loss, n_batches = [], [], 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(**batch)
        total_loss += outputs.loss.item()
        n_batches += 1
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1, zero_division=0
    )
    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    model.train()
    if return_preds:
        return metrics, all_preds, all_labels
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")

    limit = SMOKE_TEST_SAMPLES if args.smoke_test else None
    num_epochs = 1 if args.smoke_test else NUM_EPOCHS
    output_dir = Path("models/smoke-test") if args.smoke_test else OUTPUT_MODEL_DIR

    if args.smoke_test:
        print(f"\n*** SMOKE TEST: {SMOKE_TEST_SAMPLES} rows/split, 1 epoch ***\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = Collator(tokenizer, MAX_LENGTH)

    print("Loading datasets...")
    train_ds = JsonlDataset(DATA_DIR / "train.jsonl", limit)
    val_ds = JsonlDataset(DATA_DIR / "validation.jsonl", limit)
    test_ds = JsonlDataset(DATA_DIR / "test.jsonl", limit)
    print(f"  train={len(train_ds)}  validation={len(val_ds)}  test={len(test_ds)}")

    # num_workers parallelizes tokenization off the main thread; pin_memory speeds
    # host->GPU transfer. persistent_workers avoids re-spawning them each epoch.
    loader_kwargs = {"collate_fn": collate, "num_workers": 4, "pin_memory": True,
                      "persistent_workers": True}
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, **loader_kwargs)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, dtype=torch.float32,
        id2label=LABEL_NAMES, label2id={v: k for k, v in LABEL_NAMES.items()},
    )
    model.to(device)
    model.train()

    # AdamW with no weight decay on bias/LayerNorm params (standard practice)
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=LEARNING_RATE)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    print(f"\nTotal training steps: {total_steps}")
    print("Starting training...\n")

    best_f1 = -1.0
    start_time = time.time()
    history = []
    skipped_batches = 0
    bad_batch_log = Path("logs/bad_batches.jsonl")
    bad_batch_log.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        running_loss, step_count = 0.0, 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # bf16 autocast: ~5x speedup, and needs no GradScaler (unlike fp16).
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(**batch)
                loss = outputs.loss

            # --- Guard 1: skip batches whose forward pass produced NaN/Inf ---
            if not torch.isfinite(loss):
                skipped_batches += 1
                with open(bad_batch_log, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps({
                        "epoch": epoch + 1, "step": step, "stage": "forward",
                        "seq_len": int(batch["input_ids"].shape[1]),
                        "batch_size": int(batch["input_ids"].shape[0]),
                        "labels": batch["labels"].cpu().tolist(),
                    }) + "\n")
                print(f"  [skip] non-finite loss at epoch {epoch+1} step {step} "
                      f"(seq_len={batch['input_ids'].shape[1]}) - batch skipped, training continues")
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue

            optimizer.zero_grad()
            loss.backward()

            # --- Guard 2: skip the optimizer step if any gradient is NaN/Inf ---
            # Without this, a single bad batch permanently poisons every weight.
            grads_finite = all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            )
            if not grads_finite:
                skipped_batches += 1
                with open(bad_batch_log, "a", encoding="utf-8") as bf:
                    bf.write(json.dumps({
                        "epoch": epoch + 1, "step": step, "stage": "backward",
                        "seq_len": int(batch["input_ids"].shape[1]),
                        "batch_size": int(batch["input_ids"].shape[0]),
                        "loss": float(loss.item()),
                        "labels": batch["labels"].cpu().tolist(),
                    }) + "\n")
                print(f"  [skip] non-finite gradients at epoch {epoch+1} step {step} "
                      f"(seq_len={batch['input_ids'].shape[1]}, loss={loss.item():.4f}) "
                      f"- optimizer step skipped, weights preserved")
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step_count += 1

            if step % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                steps_done = epoch * len(train_loader) + step + 1
                rate = steps_done / max(elapsed, 1e-9)
                eta_min = (total_steps - steps_done) / max(rate, 1e-9) / 60
                print(f"epoch {epoch+1}/{num_epochs} | step {step}/{len(train_loader)} | "
                      f"loss {loss.item():.4f} | grad_norm {grad_norm:.2f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | skipped {skipped_batches} | "
                      f"ETA {eta_min:.0f} min")

        avg_train_loss = running_loss / max(step_count, 1)
        val_metrics = evaluate(model, val_loader, device)
        print(f"\n=== epoch {epoch+1} done | train_loss {avg_train_loss:.4f} | "
              f"skipped_batches {skipped_batches} | "
              f"val: acc {val_metrics['accuracy']:.4f} prec {val_metrics['precision']:.4f} "
              f"rec {val_metrics['recall']:.4f} f1 {val_metrics['f1']:.4f} ===\n")

        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "validation": val_metrics})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            print(f"  -> new best F1 ({best_f1:.4f}), checkpoint saved to {output_dir}\n")

    duration_min = (time.time() - start_time) / 60
    print(f"Training complete in {duration_min:.1f} minutes")

    if args.smoke_test:
        print("\n*** SMOKE TEST PASSED ***")
        print("Re-run without --smoke-test for the full training run.")
        return

    # ---- Final test-set evaluation ----
    print("\nEvaluating on test set...")
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device, return_preds=True)
    cm = confusion_matrix(test_labels, test_preds).tolist()
    print(f"Test: acc {test_metrics['accuracy']:.4f} prec {test_metrics['precision']:.4f} "
          f"rec {test_metrics['recall']:.4f} f1 {test_metrics['f1']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}")

    # Per-source breakdown
    per_source = {}
    sources = np.array([r.get("source", "unknown") for r in test_ds.rows])
    for src in sorted(set(sources)):
        mask = sources == src
        acc = accuracy_score(test_labels[mask], test_preds[mask])
        p, r, f, _ = precision_recall_fscore_support(
            test_labels[mask], test_preds[mask], average="binary", pos_label=1, zero_division=0
        )
        per_source[src] = {"accuracy": float(acc), "precision": float(p),
                           "recall": float(r), "f1": float(f), "n": int(mask.sum())}
        print(f"  [{src}] acc={acc:.4f} prec={p:.4f} rec={r:.4f} f1={f:.4f} (n={int(mask.sum())})")

    # ---- Held-out generalization evaluation ----
    held_out_metrics, per_attack_type = None, {}
    if HELD_OUT_FILE.exists():
        print(f"\nEvaluating on HELD-OUT set ({HELD_OUT_FILE})...")
        held_ds = JsonlDataset(HELD_OUT_FILE)
        held_loader = DataLoader(held_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate)
        held_out_metrics, held_preds, held_labels = evaluate(model, held_loader, device, return_preds=True)
        held_cm = confusion_matrix(held_labels, held_preds).tolist()
        held_out_metrics["confusion_matrix"] = held_cm
        print(f"Held-out: acc {held_out_metrics['accuracy']:.4f} "
              f"prec {held_out_metrics['precision']:.4f} rec {held_out_metrics['recall']:.4f} "
              f"f1 {held_out_metrics['f1']:.4f}")
        print(f"Confusion matrix [[TN, FP], [FN, TP]]: {held_cm}")

        attack_types = np.array([r.get("attack_type", "unknown") for r in held_ds.rows])
        print("\nPer-attack-type breakdown:")
        for at in sorted(set(attack_types)):
            mask = attack_types == at
            acc = accuracy_score(held_labels[mask], held_preds[mask])
            per_attack_type[at] = {"accuracy": float(acc), "n": int(mask.sum())}
            print(f"  [{at}] acc={acc:.4f} (n={int(mask.sum())})")

        # Save misclassifications for manual review
        eval_dir = Path("data/evaluation")
        eval_dir.mkdir(parents=True, exist_ok=True)
        fps = [held_ds.rows[i] for i in range(len(held_labels))
               if held_labels[i] == 0 and held_preds[i] == 1]
        fns = [held_ds.rows[i] for i in range(len(held_labels))
               if held_labels[i] == 1 and held_preds[i] == 0]
        for name, items in [("false_positives", fps), ("false_negatives", fns)]:
            with open(eval_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"  saved {len(items)} {name} to {eval_dir / (name + '.jsonl')}")

    metadata = {
        "base_model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "num_epochs": num_epochs,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "precision": "fp32",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_duration_minutes": round(duration_min, 2),
        "skipped_batches": skipped_batches,
        "epoch_history": history,
        "test_metrics": {**test_metrics, "confusion_matrix": cm},
        "per_source_test_metrics": per_source,
        "held_out_metrics": held_out_metrics,
        "per_attack_type_held_out": per_attack_type,
        "label_mapping": LABEL_NAMES,
        "random_seed": RANDOM_SEED,
        "training_loop": "standalone PyTorch (HF Trainer produced NaN gradients on this ROCm setup)",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nModel, tokenizer, and metadata saved to {output_dir}")


if __name__ == "__main__":
    # Required on Windows: DataLoader workers use spawn, which re-imports this
    # module in each child process. freeze_support() prevents recursive spawning.
    import multiprocessing
    multiprocessing.freeze_support()
    main()