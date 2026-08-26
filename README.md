# SecureRAG: Secure Middleware for RAG-Based AI Systems

**Supervised by:** Dr. Arshad Farhad

 Muhammad Abdullah Azib
**Session:** 2023-2027 | BSCS-7B

## 📌 Project Abstract

Retrieval-Augmented Generation (RAG) combines Large Language Models (LLMs) with external knowledge retrieval. However, RAG systems are susceptible to adversarial manipulation and prompt injection. SecureRAG is a robust middleware solution enforcing a "Zero Trust" architecture. It implements a multi-stage validation pipeline using AI-based threat detection (DeBERTa-v3-small), integrity verification (SHA-256), and AES-256-GCM encryption for vector database storage — running entirely locally with no cloud dependencies.

## 🔄 Project Direction Update

The project has moved from the original phase-based prototype to a **sprint-based implementation plan** (Sprint 0 → Sprint 7), which enforces a stricter build order: each sprint's backend component must be complete and verified before the next begins, and the GUI is not built until the full backend pipeline is stable. The original Phase 1–2 prototype work is preserved in `legacy/` as reference material and is not used directly in the rebuild.

## 🚀 Technologies

**Current sprint-based stack:**
- **Security classifier:** DeBERTa-v3-small (fine-tuned SAFE/MALICIOUS binary classification)
- **Embeddings:** Dedicated Sentence Transformers model
- **Vector storage:** ChromaDB
- **Integrity/confidentiality:** SHA-256 hashing, AES-256-GCM encryption
- **LLM engine:** LLaMA 3 8B via Ollama
- **Desktop interface:** PyQt6 (Sprint 6+)
- **Packaging:** PyInstaller (Sprint 7)
- **GPU acceleration:** AMD ROCm 7.14.0 (PyTorch, Windows-native)

**Legacy prototype stack (archived in `legacy/`):**
- Prototyping: Langflow
- Embeddings: Ollama Embeddings (all-minilm)
- Security Layer 1: Microsoft Presidio (PII masking), spaCy, Python `re` (rule-based filtering)

## 📂 Sprint Roadmap

| Sprint | Focus | Status |
|---|---|---|
| 0 | Environment & Foundation | ✅ Complete |
| 1 | DeBERTa Dataset & Training | ⏳ Next |
| 2 | Evaluation & Classifier API | ⏳ Pending |
| 3 | DocumentGuard (secure ingestion) | ⏳ Pending |
| 4 | Embeddings + ChromaDB + Integrity | ⏳ Pending |
| 5 | RAG + ContextGuard + LLaMA | ⏳ Pending |
| 6 | PyQt6 Desktop Application | ⏳ Pending |
| 7 | Testing + Benchmarking + Packaging | ⏳ Pending |

### ✅ Sprint 0 — Environment & Foundation (Complete)

- Verified Python 3.13.15 in a dedicated `.venv`
- Configured Git (including `safe.directory` fix for the project drive) and Antigravity IDE
- Verified AMD RX 7800 XT GPU support via native Windows ROCm 7.14.0 wheels (`torch==2.12.0+rocm7.14.0`) — required disabling Windows Smart App Control, which was blocking unsigned ROCm DLLs (`WinError 4551`)
- Built the sprint-based project skeleton (see structure below)
- Archived Phase I prototype code (`legacy/`) with a preserved Git history checkpoint (`phase1-archive` tag)

## 🗂️ Project Structure

```
SecureRAG/
├── legacy/              # Phase I prototype (Langflow, prompt_guard.py, chroma_storage) — reference only
├── assets/
├── configs/              # Thresholds, model paths, chunk size configs
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── docs/                 # Sprint notes, benchmark results, architecture docs
├── logs/                 # Audit logs, training logs
├── models/               # Trained checkpoints, tokenizer files
├── notebooks/            # Dataset inspection, training experiments
├── src/
│   ├── ingestion/        # DocumentGuard — loaders, chunking, ingestion pipeline
│   ├── llm/               # LLaMA 3 8B / Ollama integration
│   ├── rag/                # RAG orchestration, ContextGuard
│   ├── security/         # DeBERTa classifier, PromptGuard, OutputGuard
│   ├── storage/           # ChromaDB, embeddings, SHA-256/AES-256-GCM integrity
│   └── utils/
├── tests/
├── .venv/
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/azibabdullah/SecureRAG.git
   cd SecureRAG
   ```

2. Create and activate a virtual environment (Python 3.11+):
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional — AMD GPU users) Install PyTorch with ROCm support separately:
   ```
   pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1101]==2.12.0+rocm7.14.0"
   ```
   > **Note:** On Windows, this may require disabling **Smart App Control** (Windows Security → App & browser control), since AMD's ROCm DLLs are currently unsigned and get blocked with `WinError 4551`. This is a one-way setting on most Windows 11 installs — see `docs/sprint0-notes.md` for details.

5. Verify GPU setup (if applicable):
   ```
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```
