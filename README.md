# 📚 PDF Q&A Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) app. Upload any text-based PDF, ask questions, and get answers based **only** on that document, along with the pages the answer came from.

![App screenshot](screenshots/app.4.png)

## Features

- Upload any PDF and switch documents anytime
- Answers grounded in the document (says "I could not find this in the document" when the answer is missing)
- Source pages shown with every answer
- Chat-style interface with history
- Document tab: page preview and stats
- API key kept out of the code (Streamlit secrets)

## How it works

1. The PDF is loaded page by page (PyPDF)
2. Text is split into chunks (800 characters, 100 overlap)
3. Each chunk is converted to an embedding (FastEmbed, `bge-small-en-v1.5`)
4. Embeddings are stored in a FAISS index
5. The question is embedded and the top matching chunks are retrieved
6. Gemini writes the answer using only those chunks

## Tech stack

Python, Streamlit, LangChain (loaders and splitters), FastEmbed, FAISS, Google Gemini API

## Project structure

```
rag-pdf-chatbot/
├── rag_app.py            # Streamlit app
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml       # Theme colors
└── screenshots/
    └── app.png
```

## Setup

1. Clone the repository

```bash
git clone https://github.com/Laibarehman-SE/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com)

4. Create `.streamlit/secrets.toml` and add your key

```toml
GEMINI_API_KEY = "your-key-here"
```

5. Run the app

```bash
python -m streamlit run rag_app.py
```

Then open http://localhost:8501

## Limitations

- Works with text-based PDFs only (scanned or image PDFs need OCR)
- Best results with English questions and documents
- Free API tier has rate limits

## Future improvements

- Multiple PDFs in one knowledge base
- Answer quality evaluation
- Better chunking strategies
- OCR support for scanned PDFs

## Author

Laiba Rehman, Software Engineering student
