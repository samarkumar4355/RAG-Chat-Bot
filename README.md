# 📄 RAG PDF Q&A

A **Retrieval-Augmented Generation** application that lets you upload PDF documents and ask questions about their content. Answers are grounded in your documents using vector similarity search and a supported hosted LLM.

## Architecture

```
PDF Upload → Text Extraction (pypdf) → Chunking (LangChain)
         → Embedding (HuggingFace sentence-transformers, local)
         → Vector Storage (ChromaDB, local)
         → Query: Similarity Search → Context + Question → Hosted LLM → Answer
```

## Features

- **PDF Upload** — drag & drop one or more PDFs
- **Local Embeddings** — HuggingFace `all-MiniLM-L6-v2` runs on your machine (no API key needed for embeddings)
- **ChromaDB** — persistent local vector store with deduplication
- **Multiple LLM providers** — Groq, OpenRouter, or Google Gemini
- **Source Citations** — see which chunks the answer came from
- **Document Filtering** — restrict answers to a specific uploaded file
- **Chat History** — maintains conversation context within a session

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a provider API key

Create an API key with Groq, OpenRouter, or Google AI Studio.

### 3. Configure local secrets

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your keys:

```toml
GROQ_API_KEY = "your-groq-key"
OPENROUTER_API_KEY = "your-openrouter-key"
GEMINI_API_KEY = "your-gemini-key"
```

`.streamlit/secrets.toml` is ignored by Git. Never commit API keys; use newly rotated keys if any were previously checked in.

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Use

1. Select a model in the sidebar
2. Upload one or more PDF files
3. Ask questions in the chat

## Project Structure

```
website/
├── .gitignore                # Ignores local secrets and Python cache files
├── app.py                    # Streamlit UI
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .streamlit/
│   ├── config.toml           # Streamlit theme & config
│   ├── secrets.toml          # Local API keys (ignored by Git)
│   └── secrets.toml.example  # Secret configuration template
├── rag_engine/
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF loading & chunking
│   ├── vector_store.py       # ChromaDB + HuggingFace embeddings
│   └── qa_chain.py           # LangChain RAG chain + hosted LLMs
├── uploads/                  # Temp storage for uploaded PDFs
└── vectorstore/              # ChromaDB persistent storage
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk size | 500 chars | Size of text chunks for embedding |
| Chunk overlap | 100 chars | Overlap between consecutive chunks |
| Embedding model | `all-MiniLM-L6-v2` | Local HuggingFace model |
| Top-K retrieval | 5 | Number of chunks sent to the LLM |
| LLM model | `openai/gpt-oss-120b` | Default Groq model |
| Temperature | 0.2 | LLM creativity (lower = more factual) |
