"""
SecureRAG Dataset Preparation - Day 2
========================================
Merges Hlyn (direct injection) + BIPIA-70K (indirect injection, context-only
text - confirmed via manual inspection) into one training pool with
controlled oversampling, cross-source deduplication, and stratified splits.

Agentic-5K is loaded and saved SEPARATELY to data/held_out_eval/ and is
NEVER merged into the training pool - it exists purely for Day 3's
cross-dataset generalization evaluation.

Usage:
    python src/security/prepare_dataset.py
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/dataset.yaml")
PROCESSED_DIR = Path("data/processed")
HELD_OUT_DIR = Path("data/held_out_eval")
DOCS_DIR = Path("docs")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["dataset"]


def normalize_text_for_hash(text: str) -> str:
    """Collapse whitespace for duplicate detection - catches near-identical
    rows that differ only in spacing/newlines, not just byte-exact matches."""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_hlyn(source_cfg: dict) -> pd.DataFrame:
    log.info("Loading Hlyn (direct injection)...")
    ds = load_dataset(source_cfg["hf_name"])
    df = ds["train"].to_pandas()
    df = df.rename(columns={source_cfg["text_column"]: "text", source_cfg["label_column"]: "label"})
    df = df[["text", "label"]].copy()
    df["source"] = "hlyn"
    df["attack_type"] = df["label"].apply(
        lambda l: source_cfg["attack_type_default"] if l == 1 else "benign"
    )
    log.info(f"  Hlyn: {len(df)} rows")
    return df


def load_bipia(source_cfg: dict) -> pd.DataFrame:
    log.info("Loading BIPIA-70K (indirect injection)...")
    ds = load_dataset(source_cfg["hf_name"])
    df = ds["train"].to_pandas()
    # CONFIRMED: use context only, not user_intent - the injection lives in context
    df = df.rename(columns={source_cfg["text_column"]: "text", source_cfg["label_column"]: "label"})
    df = df[["text", "label"]].copy()
    df["source"] = "bipia_70k"
    df["attack_type"] = df["label"].apply(
        lambda l: source_cfg["attack_type_default"] if l == 1 else "benign_context"
    )

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    log.info(f"  BIPIA-70K: {before} -> {len(df)} rows after internal dedup ({before - len(df)} removed)")
    return df


def load_agentic_held_out(source_cfg: dict) -> pd.DataFrame:
    """Load ALL THREE splits of Agentic-5K combined, since we're not training
    on it - the authors' own train/val/test split is irrelevant to us."""
    log.info("Loading Agentic-5K (HELD OUT - not used in training)...")
    ds = load_dataset(source_cfg["hf_name"])
    frames = []
    for split_name, split_data in ds.items():
        split_df = split_data.to_pandas()
        split_df["original_split"] = split_name
        frames.append(split_df)
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns={source_cfg["text_column"]: "text", source_cfg["label_column"]: "label"})
    df["source"] = "agentic_5k"
    # Preserve the rich metadata this dataset provides - valuable for
    # detailed held-out evaluation breakdowns in Day 3
    keep_cols = ["text", "label", "source", "attack_family", "technique", "severity", "original_split"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df = df.rename(columns={"attack_family": "attack_type"})

    log.info(f"  Agentic-5K: {len(df)} rows (combined all 3 original splits)")
    return df


def oversample(df: pd.DataFrame, factor: float, seed: int) -> pd.DataFrame:
    if factor <= 1.0:
        return df
    n_repeats = int(factor)
    remainder = factor - n_repeats
    out = pd.concat([df] * n_repeats, ignore_index=True)
    if remainder > 0:
        out = pd.concat([out, df.sample(frac=remainder, random_state=seed)], ignore_index=True)
    return out


def clean(df: pd.DataFrame, prep_cfg: dict) -> pd.DataFrame:
    before = len(df)
    df["text"] = df["text"].astype(str).str.strip()
    if prep_cfg.get("remove_empty", True):
        df = df[df["text"].str.len() >= prep_cfg.get("min_text_length", 3)]
    df = df[df["text"].str.len() <= prep_cfg.get("max_text_length_chars", 80000)]
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)
    log.info(f"  Cleaning: {before} -> {len(df)} rows")
    return df.reset_index(drop=True)


def cross_source_dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df["_hash"] = df["text"].apply(lambda t: hashlib.sha256(normalize_text_for_hash(t).encode()).hexdigest())
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"]).reset_index(drop=True)
    log.info(f"Cross-source dedup: {before} -> {len(df)} rows ({before - len(df)} removed)")
    return df


def stratified_split(df: pd.DataFrame, cfg: dict):
    seed = cfg["random_seed"]
    train_df, temp_df = train_test_split(
        df, train_size=cfg["splits"]["train"], stratify=df["label"], random_state=seed
    )
    rel_val = cfg["splits"]["validation"] / (cfg["splits"]["validation"] + cfg["splits"]["test"])
    val_df, test_df = train_test_split(
        temp_df, train_size=rel_val, stratify=temp_df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_jsonl(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    log.info(f"Saved {len(df)} rows to {path}")


def main():
    cfg = load_config()
    sources = cfg["train_sources"]
    seed = cfg["random_seed"]

    # --- Load and clean each training source (NO oversampling yet) ---
    hlyn_df = clean(load_hlyn(sources["prompt_injection"]), cfg["preprocessing"])
    bipia_df = clean(load_bipia(sources["indirect_injection"]), cfg["preprocessing"])

    combined = pd.concat([hlyn_df, bipia_df], ignore_index=True)
    # Dedup FIRST, before any oversampling - oversampling-by-duplication creates
    # exact-duplicate text, so if dedup ran after oversampling it would just
    # delete the duplicates we intentionally created, silently cancelling it out.
    combined = cross_source_dedup(combined)

    # --- Statistics on the deduped (not yet oversampled) pool ---
    stats = {
        "total_samples_before_oversampling": len(combined),
        "class_distribution": combined["label"].value_counts().to_dict(),
        "source_distribution": combined["source"].value_counts().to_dict(),
        "attack_type_distribution": combined["attack_type"].value_counts().to_dict(),
        "avg_length_chars": float(combined["text"].str.len().mean()),
        "max_length_chars": int(combined["text"].str.len().max()),
    }
    log.info(f"Deduped pool stats (pre-oversampling):\n{json.dumps(stats, indent=2, default=str)}")

    # --- Split FIRST, then oversample ONLY the training split ---
    # This guarantees no duplicated row can ever appear in both train and
    # validation/test - oversampling only inflates what the model trains on,
    # never what it's evaluated against.
    train_df, val_df, test_df = stratified_split(combined, cfg)

    bipia_oversample_factor = sources["indirect_injection"]["oversample_factor"]
    train_bipia = train_df[train_df["source"] == "bipia_70k"]
    train_hlyn = train_df[train_df["source"] == "hlyn"]
    train_bipia_oversampled = oversample(train_bipia, bipia_oversample_factor, seed)
    train_df = pd.concat([train_hlyn, train_bipia_oversampled], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle

    for name, d in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        log.info(f"{name}: {len(d)} rows, source mix: {d['source'].value_counts().to_dict()}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_df, PROCESSED_DIR / "train.jsonl")
    save_jsonl(val_df, PROCESSED_DIR / "validation.jsonl")
    save_jsonl(test_df, PROCESSED_DIR / "test.jsonl")

    stats["train_rows"] = len(train_df)
    stats["validation_rows"] = len(val_df)
    stats["test_rows"] = len(test_df)
    stats["max_token_length_used"] = cfg["preprocessing"]["max_token_length"]
    with open(PROCESSED_DIR / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    # --- Held-out set: loaded, saved, NEVER merged above ---
    agentic_df = load_agentic_held_out(cfg["held_out_sources"]["agentic_5k"])
    HELD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(agentic_df, HELD_OUT_DIR / "agentic_5k.jsonl")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCS_DIR / "sprint1-day2-summary.md", "w", encoding="utf-8") as f:
        f.write("# Sprint 1 Day 2 - Dataset Preparation Summary\n\n")
        f.write(f"Training pool: {len(combined)} rows (Hlyn + BIPIA-70K, oversampled, cross-deduplicated)\n\n")
        f.write(f"Train/Val/Test: {len(train_df)} / {len(val_df)} / {len(test_df)}\n\n")
        f.write(f"Held-out (agentic_5k, NEVER trained on): {len(agentic_df)} rows\n\n")
        f.write("Agentic Prompt Injection 5K was reserved exclusively for cross-dataset "
                "generalization evaluation and was not used during model training, "
                "hyperparameter optimization, or dataset construction.\n")

    log.info("Day 2 complete.")


if __name__ == "__main__":
    main()