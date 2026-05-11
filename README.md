# SecureRAG: Secure Middleware for RAG-Based AI Systems

**Supervised by:** Dr. Arshad Farhad  
**Team:** Muhammad Abdullah Azib & Muhammad Abdullah Masood  
**Session:** 2023-2027 | BSCS-7B  

## 📌 Project Abstract
Retrieval-Augmented Generation (RAG) combines Large Language Models (LLMs) with external knowledge retrieval. However, RAG systems are susceptible to adversarial manipulation and prompt injection. SecureRAG is a robust middleware solution enforcing a "Zero Trust" architecture. It implements a dual-stage validation pipeline using AI threat detection and AES-256 encryption for vector database storage, running entirely locally.

## 🚀 Technologies Used So Far
* **LLM Engine:** Ollama (LLaMA 3 8B)
* **Prototyping:** Langflow
* **Database & Embeddings:** ChromaDB, Ollama Embeddings (all-minilm)
* **Security Layer 1:** Microsoft Presidio (PII Masking), spaCy (NLP), Python `re` (Rule-Based filtering)

## 📂 Project Phases & Checkpoints

### ✅ Phase 1: Visual Prototyping (Completed)
* **Goal:** Establish a baseline RAG pipeline locally without cloud dependencies.
* **Achievement:** Successfully built a Langflow pipeline that chunks text, generates embeddings locally via Ollama, stores them in a persistent ChromaDB (`./chroma_storage`), and retrieves context to answer queries via LLaMA 3.
* **Output:** ![Langflow Prototype](assets/image_358117.png)

### ✅ Phase 2: Layer 1 - Prompt Security (Completed)
* **Goal:** Implement the first line of defense middleware to catch jailbreaks and redact sensitive data before reaching the LLM.
* **Achievement:** Built `prompt_guard.py`. It uses a 3-step pipeline:
  1. **Rule-Based Checker:** Regex patterns to catch "ignore instructions", "developer mode", etc.
  2. **PII Masking:** Microsoft Presidio removes emails and phone numbers.
  3. **Policy Engine:** Dynamically calculates risk and outputs ALLOW, MASK, or BLOCK.
* **Output:** ![Layer 1 Architecture](assets/WhatsApp_Image_2026-05-04.jpeg)

### ⏳ Phase 3: Storage & Retrieval Layer (Next)
* **Goal:** Implement AES-256 GCM encryption for database storage and integrate Sentence Transformers natively in Python.

### ⏳ Phase 4: Context Guard & Desktop App
* **Goal:** Integrate DeBERTa-v3 for semantic anomaly detection and wrap the pipeline in a PyQt6 Desktop GUI.


## 🚀 Setup Instructions
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/SecureRAG.git](https://github.com/YOUR_USERNAME/SecureRAG.git)
   cd SecureRAG