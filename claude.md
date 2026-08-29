CLAUDE.md — SecureRAG Project Reference

This file exists so Claude Code (or any future session) has full context on what's been done, decided, and why — without re-deriving it from scratch. Update it as sprints progress.

Project Overview

SecureRAG: Secure Middleware for RAG-Based AI Systems — final year project (BSCS-7B, session 2023-2027). Supervisor: Dr. Arshad Farhad. Team: Muhammad Abdullah Azib & Muhammad Abdullah Masood.

Building a fully local, "Zero Trust" secure RAG desktop application for Windows. Core idea: a fine-tuned DeBERTa-v3-small classifier screens both user prompts (PromptGuard) and ingested document chunks (DocumentGuard) for malicious content before they reach the LLM or the vector store.

Target architecture:

Document → Extraction → Chunking → DeBERTa/DocumentGuard → Quarantine or Safe →
Embedding → ChromaDB → Retrieval → Integrity Check → ContextGuard → LLaMA 3 8B/Ollama → OutputGuard → User

Core components:

DeBERTa-v3-small — binary SAFE/MALICIOUS security classifier (shared by PromptGuard + DocumentGuard, pending Sprint 1's generalization evidence)
Sentence Transformers — dedicated embedding model (separate from DeBERTa)
ChromaDB — local vector storage
SHA-256 — integrity verification for stored chunks
AES-256-GCM — encryption for sensitive local storage
LLaMA 3 8B via Ollama — local generation
PyQt6 — Windows desktop UI (Sprint 6+)
PyInstaller — final Windows packaging (Sprint 7)

Working rule: each sprint (0–7) must be complete and verified before the next starts. GUI is not built until the backend is stable.

Legacy / Prior Work

An earlier "Phase 1–2" prototype (Langflow-based pipeline, Ollama embeddings, prompt_guard.py using Presidio + spaCy + regex) was archived to legacy/ and is reference only — not used in the current sprint-based rebuild. Git tag phase1-archive marks the preserved snapshot.

Environment (Sprint 0 — Complete)
OS/location: Windows, project root E:\SecureRAG
Python: 3.13.15, in .venv (NOT .venv — an old duplicate venv folder was deleted; always use .venv)
GPU: AMD Radeon RX 7800 XT, using native Windows ROCm 7.14.0 wheels (torch==2.12.0+rocm7.14.0, installed via --index-url https://repo.amd.com/rocm/whl-multi-arch/). NOT using WSL2/Linux — ROCm works directly on Windows now.
Known gotcha: Windows Smart App Control blocks AMD's unsigned ROCm DLLs (WinError 4551, e.g. on rocsolver.dll). Had to disable Smart App Control (Windows Security → App & browser control) — this is a one-way setting on most Windows 11 installs.
GPU verified working: torch.cuda.is_available() → True, torch.cuda.get_device_name(0) → AMD Radeon RX 7800 XT.
Git: installed, configured. Hit a "dubious ownership" error on the E: drive early on — fixed via git config --global --add safe.directory E:/SecureRAG. Remote: https://github.com/azibabdullah/SecureRAG.git.
IDE: Antigravity, interpreter pointed at .venv\Scripts\python.exe.
Repository Structure
SecureRAG/
├── legacy/                    # Phase I prototype, reference only
├── assets/
├── configs/
│   └── dataset.yaml           # Single source of truth for dataset sources/mixture/preprocessing
├── data/
│   ├── raw/
│   │   ├── prompt_injection/
│   │   └── indirect_injection/
│   ├── processed/             # train.jsonl, validation.jsonl, test.jsonl, dataset_statistics.json
│   ├── evaluation/            # false_positives.jsonl, false_negatives.jsonl (Day 3, not yet created)
│   └── held_out_eval/
│       └── agentic_5k.jsonl   # NEVER used in training — external generalization test
├── docs/                      # audit reports, sprint summaries, MODEL_CARD.md (pending)
├── logs/
├── models/
│   └── deberta-v3-small-securerag/   # trained model output (pending Day 3)
├── notebooks/
├── scripts/                   # one-off/utility scripts (not importable library code)
│   ├── audit_dataset.py
│   ├── check_token_lengths.py
│   ├── inspect_bipia_malicious.py
│   └── test_classifier.py     # Sprint 1 DONE-condition smoke test
├── src/
│   ├── ingestion/              # DocumentGuard (Sprint 3)
│   ├── llm/                    # LLaMA/Ollama integration (Sprint 5)
│   ├── rag/                    # RAG orchestration, ContextGuard (Sprint 5)
│   ├── security/               # DeBERTa classifier, PromptGuard, OutputGuard
│   │   ├── prepare_dataset.py  # Day 2 merge pipeline
│   │   └── train_deberta.py    # Day 3 training script
│   ├── storage/                # ChromaDB, embeddings, SHA-256/AES-256-GCM (Sprint 4)
│   └── utils/
│   # gui/ not yet created — add in Sprint 6
├── tests/
├── .venv/
├── .gitignore
├── requirements.txt
└── README.md
Sprint 1 — DeBERTa Security Classifier (IN PROGRESS)

Goal: train a binary SAFE(0)/MALICIOUS(1) DeBERTa-v3-small classifier that generalizes across both direct prompt injection (PromptGuard's use case) and indirect/document-embedded injection (DocumentGuard's use case), with rigorous train/eval separation.

Dataset strategy (finalized after auditing real data — do not revert to guessed ratios)
Role	Dataset	HF ID	Rows	Notes
Train — direct injection	Hlyn	hlyn/prompt-injection-judge-deberta-dataset	399,741	Already deduped by publisher. Balanced: 49.2% malicious / 50.8% benign. Columns: text, label. Redirects to hlyn-labs/... on HF — both names resolve, not an error.
Train — indirect injection	BIPIA-derived 70K	MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT	70,000	Gated — requires accepting terms on HF + huggingface-cli login. Balanced 50/50. Columns: context, user_intent, label, source. CONFIRMED via manual inspection of 5 malicious examples: use context ONLY as training text — do not concatenate user_intent. The injected instruction always lives inside context; user_intent is consistently a normal, unrelated question. This matches DocumentGuard's real input (document chunks only, no user question attached).
Held-out eval — NEVER trained on	Agentic Prompt Injection 5K	3nesdeniz/agentic-prompt-injection-5k	4,935 (3 splits combined)	Rich metadata: attack_family, technique, severity. Includes deliberate hard negatives (e.g. "explain the two-person rule for wire transfers", "summarize this phishing email"). Substitutes for Lakera's PINT benchmark, which is confirmed NOT publicly downloadable (dataset is private specifically to prevent train-time contamination; only the eval harness is open source) — do not attempt to use PINT locally.

Also considered and explicitly rejected: Hlyn V1/V2 (too small, wrong format — conversational judge/preference data, not binary classification).

Key confirmed technical decisions
max_length = 512 — confirmed via actual DeBERTa tokenizer check (not char-count proxy): 99.08% of Hlyn's tokenized length fits within 512 tokens; only 0.92% truncated.
BIPIA text field = context only (see above).
Mixture ratio target: ~75% direct / ~25% indirect in the training set, achieved via 2.0x oversampling of BIPIA (not a blind concatenation — Hlyn's raw size would otherwise dominate at ~86/14).
Cross-source dedup uses normalized SHA-256 hashing (whitespace-collapsed, lowercased) — catches near-duplicates that exact string matching misses.
Preprocessing filters: drop text <3 chars (Hlyn had a few 1-character junk rows) or >80,000 chars; drop rows with labels outside {0,1}.
Canonical schema every row is normalized to: {text, label, source, attack_type}. source/attack_type are for analysis only — not fed to the model.
Bug found and fixed (important — don't reintroduce)

Original pipeline order was wrong: oversampling BIPIA (by literal row duplication) was applied before cross-source deduplication. Since duplicating a row creates exact-duplicate text, the subsequent dedup step silently deleted almost all of the oversampled copies — the intended 25% BIPIA share collapsed back to ~14%, canceling the oversampling with no error or warning.

Fix applied: correct order is dedup → split → oversample (training split only). Oversampling must never touch validation/test, or a duplicated row could land in both train and test (leakage). This fix has been written into src/security/prepare_dataset.py; confirm on next run that BIPIA's training-split share is actually ~25%, not ~14%, before trusting the trained model's generalization numbers.

Scripts built so far
scripts/audit_dataset.py — generic HF dataset audit: schema, label distribution, duplicates, missing values, character-length stats, sample records. Reusable for any dataset by editing the call at the bottom.
scripts/check_token_lengths.py — real tokenizer-based token-length distribution (buckets: <128/128-256/256-512/>512), used to confirm max_length=512.
scripts/inspect_bipia_malicious.py — pulls malicious (label==1) BIPIA rows specifically, used to resolve the context-only decision.
scripts/test_classifier.py — Sprint 1's literal DONE-condition check: loads the trained model, runs it against a handful of obvious malicious/benign prompts.
src/security/prepare_dataset.py — Day 2 merge pipeline. Loads Hlyn + BIPIA, cleans, cross-source dedups, stratified-splits (80/10/10), oversamples training split only, saves data/processed/{train,validation,test}.jsonl + dataset_statistics.json. Separately loads and saves Agentic-5K to data/held_out_eval/agentic_5k.jsonl, structurally isolated from the training merge.
src/security/train_deberta.py — training script using HF Trainer. Reports precision/recall/F1 (not just accuracy — critical for a security classifier), breaks down test metrics by source. Currently written for CSV input from an earlier iteration — needs updating to read the .jsonl files prepare_dataset.py now produces, before Day 3 training can run.
Not yet done (Day 3 / remaining Sprint 1 work)
Update train_deberta.py to read .jsonl instead of .csv.
GPU smoke test on a small sample (1,000–5,000 rows) before full training.
Full training run (target: 2–4 epochs, per_device_train_batch_size to be tuned from VRAM headroom, metric_for_best_model="f1").
Evaluation: overall + per-source + per-attack-type precision/recall/F1, confusion matrix.
Held-out evaluation against agentic_5k.jsonl (never trained on) — this is the generalization proof point for the one-model-vs-two-models decision.
Save false positives / false negatives to data/evaluation/ and manually review ~20 of each for failure patterns (watch for hard negatives like text that discusses attacks without being one).
Decide: does one DeBERTa model generalize across direct + indirect injection adequately, or is a dedicated DocumentGuard model needed? (Decide from evidence, not upfront.)
models/deberta-v3-small-securerag/MODEL_CARD.md — datasets, hyperparameters, metrics, limitations, intended use.
docs/sprint-1.md — full sprint report.
Git commit + tag sprint-1-complete.
Sprints 2–7 (not started)

Per the original implementation plan: Sprint 2 (Evaluation & Classifier API, including generalization testing against Microsoft BIPIA proper), Sprint 3 (DocumentGuard secure ingestion), Sprint 4 (Embeddings + ChromaDB + integrity), Sprint 5 (RAG + ContextGuard + LLaMA), Sprint 6 (PyQt6 app), Sprint 7 (attack testing, benchmarking, packaging).

Open/Unresolved Items
main.py exists at project root — origin/purpose unconfirmed; likely intended as the eventual PyQt6 entry point (Sprint 6). Don't let it become an unplanned dumping ground.
GitHub contributions graph wasn't showing commits at one point — likely a commit-email mismatch (git log -1 --pretty=format:"%an <%ae>" vs. verified GitHub account email). Not confirmed resolved.