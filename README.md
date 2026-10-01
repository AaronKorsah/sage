# SAGE — Document Q&A

A Retrieval-Augmented Generation (RAG) system that lets you upload PDFs and ask questions about them.

🔗 **[Live Demo](https://sage-5iof.onrender.com)** — try it now (first load takes ~30 sec due to free-tier cold start)

## Features

- Upload PDF documents via a web interface
- Ask natural-language questions
- Answers grounded in your documents (no hallucination)
- Multi-document support with per-document filtering
- Local embedding model (no external embedding API needed)
- Fallback LLM (Groq) when the primary (Gemini) is unavailable

## Tech Stack

- **Backend**: FastAPI
- **Embeddings**: sentence-transformers (BAAI/bge-small-en-v1.5)
- **Vector store**: ChromaDB
- **LLM**: Google Gemini (primary), Groq (fallback)
- **PDF parsing**: pypdf
- **Frontend**: HTML + CSS + vanilla JavaScript

## Architecture

PDF → read_pdf() → smart_chunk() → embed() → ChromaDB
                                                    ↓
Question → embed() → retrieve() → build_prompt() → LLM → answer

## Setup

1. Create a `.env` file with:
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key

2. Install dependencies:
pip install -r requirements.txt

3. Run:
python -m uvicorn app:app --reload

4. Open http://127.0.0.1:8000

## License

MIT