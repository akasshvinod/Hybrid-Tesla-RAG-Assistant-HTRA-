# 🚗 **Tesla Model 3 Hybrid RAG Assistant**  
**AI & Automation Internship Selection Challenge — Option 1**

---

## ✅ **Task Chosen**
I selected **Option 1: Hybrid Support Bot (RAG System)** from the challenge instructions.  
This task requires building a Retrieval-Augmented Generation (RAG) assistant that answers questions strictly from a document — in this case, the **Tesla Model 3 Owner’s Manual** — with no hallucinations or external knowledge.

---

## 📘 **Project Overview**
This project implements a fully offline, metadata-aware RAG system that:

- Parses the **Tesla Model 3 Owner’s Manual (PDF)**  
  https://www.tesla.com/ownersmanual/model3/en_us/Owners_Manual.pdf
- Extracts **chapters, headings, page numbers, and metadata**
- Cleans and chunks the manual for efficient retrieval  
- Embeds content using **nomic-embed-text** (via Ollama)
- Stores vectors inside a **persistent ChromaDB** database
- Retrieves relevant sections using **hybrid (vector + metadata) search**
- Builds a strict **grounded RAG prompt** with hallucination prevention
- Generates responses using **llama3.1:8b-instruct-q4_K_M** locally
- Provides a **CLI assistant** and a **Streamlit UI**

### ⭐ This project fulfills all requirements for **Option 1**.

---

### ⚙️ How to Set Up & Run the Project
#### 1️⃣ Clone the Repository

git clone https://github.com/akasshvinod/Hybrid-Tesla-RAG-Assistant-HTRA-.git

#### 2️⃣ Create Environment
conda create -n ragbot python=3.10 -y
conda activate ragbot

#### 3️⃣ Install Dependencies
pip install -r requirements.txt

#### 4️⃣ Pull Required Models (Ollama)
ollama pull llama3.1
ollama pull nomic-embed-text

#### 📘 Step 1 — Ingest the Manual

Converts the PDF → cleaned text → chunks → embeddings → Chroma vector store.

python -m src.ingestion.ingest

#### 🔍 Step 2 — Test Retrieval
python -m src.retrieval.retriever

#### 🤖 Step 3 — Run RAG Pipeline (CLI)
python -m src.pipeline.rag_pipeline

#### 🖥️ Step 4 — Launch Streamlit Web Interface
streamlit run app/app.py

The UI includes:

Chat interface

Retrieved context preview

Latency metrics

Conversation memory

Optional chapter filter

### 📚 Why These Libraries & Models?
- LangChain 2025

- Modern LCEL pipelines

- Clean modular RAG orchestration

- ChromaDB

- Fast, persistent local vector store

- PyMuPDF

- Accurate PDF parsing for structured manuals

- Ollama + Llama 3.1

- Fully offline inference

- No API cost

- High grounding accuracy

- nomic-embed-text

- High-quality embeddings designed for documents

- Streamlit

- Quick and interactive UI

## 🎥 Demo Video
Here is the demonstration video for the Tesla Model 3 Hybrid RAG Assistant:

🔗 https://www.loom.com/share/48e669a811b3458abf6c472efdb9500f

