# 🌍 Multi-Country Content Q&A Agent

### AI Engineer Interview — One-Day Build Prototype

This repository implements a **grounded Retrieval-Augmented Generation (RAG) system** for a multi-country B2B platform.
It ensures **strict multi-tenant isolation** and provides **verifiable, citation-backed answers** using an agentic workflow powered by LangGraph.

---

## 🏗️ Architecture

The system follows a modular, agent-based pipeline:

### 🔹 Ingestion

* Processes raw JSONL content
* Uses `sentence-transformers` for embeddings
* Stores vectors in a local **FAISS index**
* Attaches rich metadata (country, language, content_id)

### 🔹 Retrieval

* Applies **strict metadata filtering** (Country + Language)
* Prevents cross-tenant data leakage
* Only relevant scoped documents are retrieved

### 🔹 Generation

* Uses **Llama 3 (via Groq API)**
* Generates answers strictly from retrieved context
* Ensures grounded and factual responses

---

## 🚀 Setup & Installation

### 1️⃣ Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2️⃣ Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### 3️⃣ Data Ingestion

Build the vector store:

```bash
python ingest.py
```

---

### 4️⃣ Start API Server

```bash
python main.py
```

Server will run at:

```
http://127.0.0.1:8000
```

Swagger docs available at:

```
/docs
```

---

## 🔒 Multi-Tenant Scoping

This system enforces **strict isolation between tenants**:

### ✅ Pre-Retrieval Filtering

* Queries FAISS with metadata constraints
* Ensures only relevant country/language data is retrieved

### ✅ Zero Data Leakage

* No cross-country content exposure
* Hard filter applied before similarity search

### ✅ Citation Fidelity

* Every response includes:

  * `content_id`
  * raw text excerpt
* Enables full traceability

---

## 🧪 Example API Request

```bash
curl -X POST http://127.0.0.1:8000/ask \
-H "Content-Type: application/json" \
-d '{
  "question": "What is the return policy?",
  "country": "A",
  "language": "en"
}'
```

---

## 📦 Tech Stack

* **LangGraph** — Agent orchestration
* **FAISS** — Vector database
* **Sentence Transformers** — Embeddings
* **Groq (Llama 3)** — LLM inference
* **FastAPI** — API layer

---

## ✅ Submission Checklist

* [x] LangGraph agent orchestration
* [x] Multi-tenant isolation (Country + Language)
* [x] Grounded responses with citations
* [x] FastAPI evaluation-ready endpoint
* [x] Local vector database (FAISS)

---

## 🎯 Key Highlights

* Production-style **multi-tenant safety**
* Fully **traceable RAG pipeline**
* Clean **agent-based architecture**
* Built within **1-day constraint**

---

## 🚀 Future Improvements

* Add hybrid search (BM25 + vector)
* Implement reranking (cross-encoder)
* Add streaming responses
* Deploy using Docker + cloud infra

---

## 👨‍💻 Author

**Prasad Mukkawar**
AI Engineer | GenAI | NLP | RAG Systems
