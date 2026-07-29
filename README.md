# Nabu — AI-Powered Bookstore Platform

**Result:** 3rd place, FinTech Hackathon by Makers | Zain Iraq (Oct 2025)

Nabu is an AI-powered book discovery platform that uses semantic search and
large language model embeddings to help users find books matching a natural-
language description of what they want to read — filterable by category and
emotional tone.

## Screenshots

![Dashboard](drag-dashboard-screenshot-here.png)
![Search results](drag-results-screenshot-here.png)

## How it works

1. **Data pipeline** (`Data-Exploration.ipynb`) — loads and cleans a 7k-book
   metadata dataset (Kaggle).
2. **Text classification** (`text-classification.ipynb`) — assigns simplified
   genre categories to each book.
3. **Sentiment analysis** (`sentiment-analysis.ipynb`) — scores each book's
   description across emotional tones (joy, surprise, anger, fear, sadness),
   enabling tone-based filtering (e.g. "show me something suspenseful").
4. **Vector search** (`vector-search.ipynb`) — generates tagged descriptions
   and embeds them for semantic similarity search.
5. **Dashboard** (`gradio-dashboard.py`) — the user-facing app. Takes a
   free-text query (e.g. "a story about forgiveness"), retrieves semantically
   similar books via a Chroma vector store, filters by category/tone, and
   displays results with cover art and generated captions.

## Stack

Python, LangChain, Hugging Face Embeddings (`sentence-transformers/all-MiniLM-L6-v2`),
ChromaDB (vector store), Gradio, Pandas, PyTorch (via Hugging Face inference)

## Setup

```bash
pip install pandas numpy python-dotenv langchain langchain-community gradio chromadb sentence-transformers
```

Run the notebooks in order (Data-Exploration → text-classification →
sentiment-analysis → vector-search) to regenerate the processed CSV/text
files, then launch the app:

```bash
python gradio-dashboard.py
```

## Notes

- The `chroma_books/` vector database folder is regenerated automatically on
  first run and is excluded from version control (see `.gitignore`).
- No API keys are required for the core recommendation pipeline (Hugging
  Face embeddings run locally); if you extend this with an LLM-based
  re-ranking step, add your key to a local `.env` file (never commit it).
