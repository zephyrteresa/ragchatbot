# Multimodal PDF Question Answering System

A Python-based Retrieval-Augmented Generation (RAG) tool for searching and asking questions about educational PDF documents containing both text and images.

---

## What It Does

- **Extracts Text & Figures:** Reads PDFs page-by-page and extracts embedded images.
- **Describes Images:** Uses vision models to summarize figures, charts, and diagrams into searchable text.
- **Local Storage:** Stores document chunks and image descriptions locally in a ChromaDB vector database.
- **Query Cleaning:** Refines user follow-up questions before searching to pull the most relevant pages.
- **Grounded Answers:** Answers questions strictly using facts found in the retrieved PDF chunks.

---

## Project Structure

```text
├── app.py                 # Main entry point and CLI runner
├── vectordb.py            # Local ChromaDB vector store wrapper
├── config/
│   ├── config.yaml        # Search threshold and chat settings
│   └── prompt_config.yaml # Prompts for rewriting and answering
├── data/                  # Place sample PDF files here
└── requirements.txt       # Python dependencies
```

---

## Setup & Installation

1. **Clone the repository and set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Copy `.env.example` to `.env` and add API credentials:
   ```bash
   cp .env.example .env
   ```
   *Required settings inside `.env`:*
   ```env
   GROQ_API_KEY=your_groq_api_key
   VISION_MODEL=qwen/qwen3.8-27b
   GROQ_MODEL=openai/gpt-oss-20b
   HF_TOKEN=your_huggingface_token
   ```

---

## How to Use

1. Place target PDFs into the `data/` folder.
2. Start the interactive assistant:
   ```bash
   python app.py
   ```
   *The system automatically parses PDFs on startup, generates image summaries, embeds text chunks, and saves them locally into ChromaDB.*
3. Ask questions in the terminal prompt. Type `quit` to exit.