"""
Visualize SecureRAG training results in the terminal.

Reads models/deberta-v3-small-securerag/training_metadata.json and renders:
  1. Per-epoch validation metrics (accuracy / precision / recall / F1)
  2. Training loss curve
  3. In-distribution vs held-out comparison
  4. Per-attack-type accuracy on the held-out set
  5. Confusion matrices

Charts use plotext when available. plotext's API differs between major
versions, so every call is probed defensively - if a chart fails for any
reason, that section falls back to ASCII bars instead of crashing. The
numeric tables always print regardless.

Run:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --ascii     # skip plotext entirely
    python scripts/visualize_results.py --probe     # list plotext's API
"""

import argparse
import json
from pathlib import Path

METADATA_PATH = Path("models/deberta-v3-small-securerag/training_metadata.json")

try:
    import plotext as _plt
    HAS_PLOTEXT = True
except ImportError:
    _plt = None
    HAS_PLOTEXT = False

USE_CHARTS = HAS_PLOTEXT
_chart_warning_shown = False


# ---------------------------------------------------------------- plotext shims

def _call_first(names, *args, **kwargs):
    """Call the first attribute in `names` that exists on plotext.
    Returns True if one was called, False if none exist."""
    for name in names:
        fn = getattr(_plt, name, None)
        if callable(fn):
            fn(*args, **kwargs)
            return True
    return False


def _reset():
    # v5 used clear_figure(); v6 renamed/reorganised these
    _call_first(["clear_figure", "clf", "clear_data", "clear"])


def _size(w, h):
    _call_first(["plotsize", "plot_size"], w, h)


def _theme(name="pro"):
    try:
        _call_first(["theme"], name)
    except Exception:
        pass  # themes are cosmetic - never worth failing over


def _show():
    _call_first(["show"])


def _note_chart_failure(err):
    global _chart_warning_shown
    if not _chart_warning_shown:
        print(f"\n  (plotext charts unavailable - {type(err).__name__}: {err}."
              f"\n   Falling back to ASCII bars. Run with --probe to inspect the API,"
              f"\n   or --ascii to skip plotext entirely.)")
        _chart_warning_shown = True


# ---------------------------------------------------------------- ASCII output

def ascii_bar(label, value, max_value=1.0, width=44, label_width=30):
    frac = 0.0 if max_value == 0 else max(0.0, min(1.0, value / max_value))
    filled = int(frac * width)
    bar = "#" * filled + "." * (width - filled)
    return f"  {label:<{label_width}} |{bar}| {value:.4f}"


def section(title):
    print()
    print("=" * 84)
    print(f"  {title}")
    print("=" * 84)


def confusion_matrix_table(cm, title):
    """cm is [[TN, FP], [FN, TP]]"""
    (tn, fp), (fn, tp) = cm
    total = tn + fp + fn + tp
    print(f"\n  {title}")
    print("                      Predicted SAFE   Predicted MALICIOUS")
    print(f"    Actual SAFE       {tn:>14,}   {fp:>19,}")
    print(f"    Actual MALICIOUS  {fn:>14,}   {tp:>19,}")
    print(f"\n    False positives: {fp:,} ({fp / max(tn + fp, 1) * 100:.1f}% of benign)")
    print(f"    False negatives: {fn:,} ({fn / max(fn + tp, 1) * 100:.1f}% of malicious)")
    print(f"    Total evaluated: {total:,}")


# ---------------------------------------------------------------- sections

def plot_epoch_metrics(history):
    epochs = [h["epoch"] for h in history]
    metrics = {
        "accuracy": [h["validation"]["accuracy"] for h in history],
        "precision": [h["validation"]["precision"] for h in history],
        "recall": [h["validation"]["recall"] for h in history],
        "f1": [h["validation"]["f1"] for h in history],
    }

    section("VALIDATION METRICS BY EPOCH")

    charted = False
    if USE_CHARTS:
        try:
            _reset()
            for name, values in metrics.items():
                _plt.plot(epochs, values, label=name)
            _plt.title("Validation metrics per epoch")
            _size(80, 22)
            _theme()
            _show()
            charted = True
        except Exception as e:
            _note_chart_failure(e)

    if not charted:
        for name, values in metrics.items():
            print(f"\n  {name.upper()}")
            for e, v in zip(epochs, values):
                print(ascii_bar(f"epoch {e}", v, 1.0, label_width=12))

    print("\n  epoch | train_loss | accuracy | precision |  recall  |    F1")
    print("  ------+------------+----------+-----------+----------+---------")
    for h in history:
        v = h["validation"]
        print(f"  {h['epoch']:>5} | {h['train_loss']:>10.4f} | {v['accuracy']:>8.4f} | "
              f"{v['precision']:>9.4f} | {v['recall']:>8.4f} | {v['f1']:>7.4f}")


def plot_train_loss(history):
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]

    section("TRAINING LOSS")

    charted = False
    if USE_CHARTS:
        try:
            _reset()
            _plt.plot(epochs, losses)
            _plt.title("Training loss per epoch")
            _size(80, 18)
            _theme()
            _show()
            charted = True
        except Exception as e:
            _note_chart_failure(e)

    if not charted:
        max_loss = max(losses) if losses else 1.0
        for e, l in zip(epochs, losses):
            print(ascii_bar(f"epoch {e}", l, max_loss, label_width=12))


def plot_distribution_comparison(test_metrics, held_out_metrics):
    section("IN-DISTRIBUTION vs HELD-OUT  (the generalization gap)")

    names = ["accuracy", "precision", "recall", "f1"]
    in_dist = [test_metrics[n] for n in names]
    held = [held_out_metrics[n] for n in names] if held_out_metrics else [0.0] * 4

    charted = False
    if USE_CHARTS:
        try:
            _reset()
            _plt.multiple_bar(names, [in_dist, held],
                              labels=["in-distribution", "held-out"])
            _plt.title("In-distribution vs held-out")
            _size(80, 22)
            _theme()
            _show()
            charted = True
        except Exception as e:
            _note_chart_failure(e)

    if not charted:
        print("\n  IN-DISTRIBUTION (test set)")
        for n, v in zip(names, in_dist):
            print(ascii_bar(n, v, 1.0, label_width=12))
        print("\n  HELD-OUT (agentic-5k, never trained on)")
        for n, v in zip(names, held):
            print(ascii_bar(n, v, 1.0, label_width=12))

    print("\n  metric    | in-dist | held-out |  gap")
    print("  ----------+---------+----------+--------")
    for n, a, b in zip(names, in_dist, held):
        print(f"  {n:<9} | {a:>7.4f} | {b:>8.4f} | {a - b:>+6.4f}")


def plot_per_source(per_source):
    if not per_source:
        return
    section("PER-SOURCE METRICS (in-distribution test set)")

    sources = sorted(per_source.keys())

    if USE_CHARTS:
        try:
            _reset()
            series = [[per_source[s][m] for s in sources]
                      for m in ("accuracy", "precision", "recall", "f1")]
            _plt.multiple_bar(sources, series,
                              labels=["accuracy", "precision", "recall", "f1"])
            _plt.title("Metrics by training source")
            _size(80, 20)
            _theme()
            _show()
        except Exception as e:
            _note_chart_failure(e)

    print("\n  source        | accuracy | precision |  recall  |    F1    |    n")
    print("  --------------+----------+-----------+----------+----------+---------")
    for s in sources:
        d = per_source[s]
        print(f"  {s:<13} | {d['accuracy']:>8.4f} | {d['precision']:>9.4f} | "
              f"{d['recall']:>8.4f} | {d['f1']:>8.4f} | {d['n']:>7,}")


def plot_per_attack_type(per_attack):
    if not per_attack:
        return
    section("PER-ATTACK-TYPE ACCURACY (held-out set)")

    items = sorted(per_attack.items(), key=lambda kv: kv[1]["accuracy"])
    labels = [k for k, _ in items]
    values = [v["accuracy"] for _, v in items]

    charted = False
    if USE_CHARTS:
        try:
            _reset()
            _plt.bar(labels, values, orientation="horizontal")
            _plt.title("Held-out accuracy by attack type")
            _size(84, 18)
            _theme()
            _show()
            charted = True
        except Exception as e:
            _note_chart_failure(e)

    if not charted:
        for label, value in zip(labels, values):
            print(ascii_bar(label, value, 1.0))

    print("\n  attack type                     | accuracy |    n")
    print("  --------------------------------+----------+--------")
    for k, v in items:
        marker = "  <-- worst" if v["accuracy"] == values[0] else ""
        print(f"  {k:<31} | {v['accuracy']:>8.4f} | {v['n']:>6,}{marker}")

    print("\n  (0.50 = chance level for binary classification)")


def probe():
    """Print plotext's public API so the shims above can be adjusted if needed."""
    if not HAS_PLOTEXT:
        print("plotext is not installed.")
        return
    version = getattr(_plt, "__version__", "unknown")
    print(f"plotext version: {version}\n")
    names = sorted(n for n in dir(_plt) if not n.startswith("_"))
    for i in range(0, len(names), 4):
        print("  " + "".join(f"{n:<26}" for n in names[i:i + 4]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ascii", action="store_true", help="Skip plotext, use ASCII bars only")
    parser.add_argument("--probe", action="store_true", help="Print plotext's available functions")
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    global USE_CHARTS
    if args.ascii:
        USE_CHARTS = False

    if not METADATA_PATH.exists():
        print(f"Could not find {METADATA_PATH}")
        print("Run training first, or adjust METADATA_PATH at the top of this script.")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print()
    print("#" * 84)
    print("#  SecureRAG - DeBERTa-v3-small Security Classifier - Training Results")
    print("#" * 84)
    print(f"  base model      : {meta.get('base_model')}")
    print(f"  epochs          : {meta.get('num_epochs')}")
    print(f"  batch size      : {meta.get('train_batch_size')}")
    print(f"  learning rate   : {meta.get('learning_rate')}")
    print(f"  max length      : {meta.get('max_length')}")
    print(f"  precision       : {meta.get('precision')}")
    print(f"  GPU             : {meta.get('gpu')}")
    print(f"  duration        : {meta.get('training_duration_minutes')} min")
    print(f"  skipped batches : {meta.get('skipped_batches')}")

    if not HAS_PLOTEXT:
        print("\n  (plotext not installed - using ASCII bars. pip install plotext)")

    history = meta.get("epoch_history", [])
    if history:
        plot_epoch_metrics(history)
        plot_train_loss(history)

    test_metrics = meta.get("test_metrics", {})
    held_out = meta.get("held_out_metrics")
    if test_metrics:
        plot_distribution_comparison(test_metrics, held_out)

    plot_per_source(meta.get("per_source_test_metrics"))
    plot_per_attack_type(meta.get("per_attack_type_held_out"))

    section("CONFUSION MATRICES")
    if test_metrics.get("confusion_matrix"):
        confusion_matrix_table(test_metrics["confusion_matrix"], "In-distribution (test set)")
    if held_out and held_out.get("confusion_matrix"):
        confusion_matrix_table(held_out["confusion_matrix"], "Held-out (agentic-5k)")

    print()


if __name__ == "__main__":
    main()