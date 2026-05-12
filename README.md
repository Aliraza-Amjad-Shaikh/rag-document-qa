# 📚 RAG-Powered Document Q&A System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2.1-green?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you
upload PDF documents and images, ask natural language questions, and receive
accurate answers with **exact source citations**, **confidence scoring**, and a
**smart "I don't know" fallback** to prevent hallucination.

---

## 📸 Demo Screenshots

> _Upload your documents and start asking questions instantly._

| Document Upload | Q&A with Citations |
|---|---|
| ![Upload Screen](screenshots/upload.png) | ![QA Screen](screenshots/qa.png) |


---

## 🏗️ Architecture

```mermaid
┌─────────────────────────────────────────────────────────────┐
│ USER INTERFACE │
│ Streamlit (app.py) │
└──────────────┬──────────────────────────┬───────────────────┘
│ │
▼ ▼
┌─────────────────────────┐ ┌────────────────────────────────┐
│ INGESTION PIPELINE │ │ Q&A PIPELINE │
│ ingestion.py │ │ │
│ │ │ 1. User asks question │
│ PDF Files │ │ 2. Embed question │
│ └─ PyMuPDF extracts │ │ 3. FAISS similarity search │
│ text (multi-mode) │ │ 4. Compute confidence score │
│ └─ Text cleaned │ │ 5. Route to LLM or fallback │
│ └─ Chunked (1000 chars)│ │ 6. GPT-4o generates answer │
│ │ │ 7. Return answer + citations │
│ Image Files │ └────────────────────────────────┘
│ └─ GPT-4o Vision │
│ describes content │ ┌────────────────────────────────┐
│ └─ Treated as text │ │ CONFIDENCE SCORING │
│ │ │ │
│ All chunks get: │ │ High (≥ 0.60) 🟢 │
│ - source filename │ │ Medium (≥ 0.40) 🟡 │
│ - page number │ │ Low (< 0.40) 🔴 │
│ - chunk ID │ │ │
│ - document type │ │ Low → "I don't know" │
└──────────┬──────────────┘ └────────────────────────────────┘
│
▼
┌─────────────────────────┐
│ VECTOR STORE │
│ retrieval.py │
│ │
│ OpenAI Embeddings │
│ text-embedding-3-small │
│ ↓ │
│ FAISS Local Index │
│ (Persistent on disk) │
└─────────────────────────┘
```

---

## ✨ Features

- 📄 **PDF Ingestion** — Multi-mode text extraction using PyMuPDF (handles columns, tables, complex layouts)
- 🖼️ **Image Understanding** — GPT-4o Vision extracts text, describes charts, tables, and diagrams
- 🔍 **Semantic Search** — FAISS vector store with OpenAI embeddings for accurate retrieval
- 📊 **Confidence Scoring** — Every answer is scored High/Medium/Low based on retrieval similarity
- 🚫 **Hallucination Prevention** — Strict "I don't know" fallback when context is insufficient
- 📚 **Source Citations** — Every claim is cited with exact filename and page number
- 💾 **Persistent Index** — Vector store saved to disk, no re-embedding on restart
- 🎨 **Modern UI** — Clean Streamlit interface with chat-style Q&A and color-coded confidence badges
- 🔄 **Multi-file Support** — Upload and query multiple PDFs and images simultaneously

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM (Chat + Vision) | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (local) |
| PDF Extraction | PyMuPDF (fitz) |
| RAG Orchestration | LangChain 0.2 |
| Image Processing | Pillow |
| Environment | python-dotenv |

---

## 📁 Project Structure

    rag-document-qa/
    │
    ├── app.py # Streamlit frontend — chat UI, file upload, display logic
    ├── ingestion.py # Document ingestion — PDF extraction + image vision pipeline
    ├── retrieval.py # FAISS vector store — embed, store, retrieve, confidence score
    ├── generation.py # LLM answer generation — prompt engineering, source citing
    ├── config.py # Central configuration — paths, models, thresholds
    │
    ├── uploads/ # Temporary storage for uploaded files (git-ignored)
    ├── vectorstore/ # Persistent FAISS index (git-ignored)
    ├── utils/
    │ └── init.py # Helper utilities
    │
    ├── .env # API keys (git-ignored — never commit this)
    ├── .gitignore # Git ignore rules
    ├── requirements.txt # Python dependencies
    └── README.md # This file



---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- An OpenAI API key ([get one here](https://platform.openai.com))
- Git installed

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa
2. Create Virtual Environment
```
```Bash

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
# Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies

```Bash

pip install -r requirements.txt
4. Configure Environment
Create a .env file in the project root:

env

OPENAI_API_KEY=your-actual-api-key-here
⚠️ Never commit your .env file. It is already in .gitignore.
```
5. Run the Application
```Bash

streamlit run app.py
The app will open automatically at http://localhost:8501
```

💡 How to Use
Upload Documents

Click "Browse files" in the sidebar
Select one or more PDF files and/or images (JPG, PNG, WEBP)
Click "🚀 Process & Index" and wait for indexing to complete
Ask Questions

Type your question in the chat input at the bottom
Press Enter to submit
Read the Answer

The answer appears with source citations
A confidence badge shows 🟢 High / 🟡 Medium / 🔴 Low
Sources show the exact filename and page number
Reset

Click "🗑️ Clear & Upload New" to start fresh with new documents
⚙️ Configuration
All settings are in config.py:

Setting	Default	Description
CHUNK_SIZE	1000	Characters per text chunk
CHUNK_OVERLAP	150	Overlap between chunks
TOP_K_RESULTS	6	Number of chunks retrieved per query
CONFIDENCE_HIGH	0.60	Threshold for High confidence
CONFIDENCE_MEDIUM	0.40	Threshold for Medium confidence
VISION_MODEL	gpt-4o	Model used for image understanding
CHAT_MODEL	gpt-4o	Model used for answer generation
EMBEDDING_MODEL	text-embedding-3-small	Model used for embeddings
⚠️ Limitations
Scanned PDFs — PDFs that are scanned images (not text-based) cannot be processed by PyMuPDF. Use image upload instead for scanned documents.
Large Files — Very large PDFs (100+ pages) may take significant time to embed and will consume OpenAI API credits.
Image Size — Images are compressed to under 3MB before sending to GPT-4o Vision. Very low-quality images may produce poor descriptions.
Context Window — Answers are based only on retrieved chunks (top-6), not the entire document. Very broad questions like "summarize this document" may return incomplete answers.
Language — Best performance on English documents. Other languages may work but are not tested.
No Memory — Each question is answered independently. The system does not maintain conversation history across questions.
API Costs — Every question and every image upload consumes OpenAI API credits. Monitor your usage at platform.openai.com.
🔮 Future Improvements
 Support for scanned PDFs using OCR (Tesseract / AWS Textract)
 Document summarization mode
 Conversation memory (multi-turn Q&A)
 Support for DOCX, PPTX, CSV file formats
 Local LLM support (Ollama / LLaMA)
 User authentication
 Answer export (PDF / Word)
 Batch question processing
📄 License
This project is licensed under the MIT License.

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
🙏 Acknowledgements
LangChain — RAG orchestration framework
OpenAI — GPT-4o and embedding models
FAISS — Facebook AI Similarity Search
PyMuPDF — PDF text extraction
Streamlit — Frontend framework

<p align="center">Built with ❤️ using LangChain + OpenAI + FAISS + Streamlit by Aliraza Amjad Shaikh</p> ```
