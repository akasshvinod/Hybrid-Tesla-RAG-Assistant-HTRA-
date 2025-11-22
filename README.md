# 🚗 Tesla Model 3 Hybrid RAG Assistant
**AI & Automation Internship Selection Challenge — Option 1**

---

## ✅ Task Chosen
I selected **Option 1: Hybrid Support Bot (RAG System)** from the challenge instructions.  
This task requires building a Retrieval-Augmented Generation (RAG) assistant that answers questions strictly based on a provided document — in this case, the **Tesla Model 3 Owner’s Manual** — without hallucinating or adding external information.

---

## 📘 Project Overview
This RAG system:

- Parses the Tesla Model 3 Owner’s Manual (PDF) 
  https://www.tesla.com/ownersmanual/model3/en_us/Owners_Manual.pdf
- Extracts headings, chapters, and metadata
- Chunks and embeds content using **nomic-embed-text** (via Ollama)
- Stores vectors inside a **ChromaDB** persistent database
- Retrieves the most relevant sections for each query
- Builds a strict, grounded RAG prompt with hallucination prevention
- Generates answers using **llama3.1:8b-instruct-q4_K_M** (local)
- Provides both a **CLI interface** and a **Streamlit web UI**

##### This project fulfills all requirements for Option 1.

---

## 📁 Project Structure

project/
│
├── data/
│   └── ec2-ug.pdf
│
├── src/
│   ├── parsing/
│   │   ├── pdf_reader.py
│   │   ├── heading_extractor.py
│   │   └── text_cleaner.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   └── ingest.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── llm/
│   │   ├── llama_client.py
│   │   └── prompts.py
│   │
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   │
│   ├── api/
│   │   └── cli_query.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── types.py
│       └── config.py
│
├── chroma_db/
│
│
├── demo/
│   ├── screenshots/
│   └── demo_video.mp4
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
├── .env
└── .gitignore



## ⚙️ How to Set Up & Run the Project

#### 1️⃣ Clone the Repository

git clone https://github.com/YOUR_USERNAME/hybrid_rag_bot.git

#### 2️⃣ Create Environment

conda create -n ragbot python=3.10 -y
conda activate ragbot

#### 3️⃣ Install Dependencies
pip install -r requirements.txt

#### 4️⃣ Pull Required Models (Ollama)
ollama pull llama3.1
ollama pull nomic-embed-text

### 📘 Step 1 — Ingest the Manual

Converts the PDF → clean text → chunks → embeddings → Chroma vector store.

python -m src.ingestion.ingest

### 🔍 Step 2 — Test Retrieval
python -m src.retrieval.retriever

### 🤖 Step 3 — Run RAG Pipeline (CLI)
python -m src.pipeline.rag_pipeline

### 🖥️ Step 4 — Launch Streamlit Web Interface
streamlit run app/app.py


#### The UI includes:

- Chat interface

- Retrieved context preview

- Latency metrics

- Conversation memory

- Optional chapter filter

### 📚 Why These Libraries & Models?

- LangChain 2025

- Used for modern, modular RAG orchestration (LCEL pipelines, prompt templates, retrieval flows).

- ChromaDB

- Lightweight, persistent vector store ideal for local/manual-based retrieval.

- PyMuPDF

- Reliable PDF extraction needed for structured parsing of the Tesla manual.

- Ollama + Llama 3.1

- Local LLM with no API cost, optimized for grounded question answering.

- nomic-embed-text

- Fast, high-quality embeddings suited for large manuals and technical documents.

- Streamlit

- Makes the assistant easy to demo and interact with through a web UI.

## 📸 Screenshots

### Ingestion Completed
![ingestion](screenshots/ingestion1.png)
![ingestion](screenshots/ingestion2.png)
![ingestion](screenshots/ingestion3.png)

### Retriever Output
![retriever](screenshots/retriever.png)

### RAG Pipeline Answer
![rag](screenshots/rag_pipeline.png)

### CLI UI
![cli](screenshots/cli.png)

### Streamlit UI
![ui](screenshots/streamlit_ui.png)
