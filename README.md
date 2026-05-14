```markdown
# 📚 RAG Document Q&A System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2.1-green?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5%20Flash-4285F4?style=for-the-badge&logo=google)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you
upload PDF documents and images, ask natural language questions, and receive
accurate answers with **exact source citations**, **confidence scoring**, and a
**smart "I don't know" fallback** to prevent hallucination.

Powered by **Google Gemini 2.5 Flash** for LLM + Vision, **HuggingFace Inference API**
for embeddings, **LangChain Semantic Chunking** for intelligent document splitting,
and **FAISS** for fast local vector search.

---

## 📸 Demo Screenshots

> Upload your documents and start asking questions instantly.

| Document Upload | Q&A with Citations |
|---|---|
| ![Upload Screen](screenshots/upload.png) | ![QA Screen](screenshots/qa.png) |


---


## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT FRONTEND                           │
│                             app.py                                   │
│                                                                      │
│   Sidebar                          Main Chat Area                    │
│   ├── File Uploader                ├── Chat History                  │
│   ├── Process & Index Button       ├── User Bubbles                  │
│   ├── System Status                ├── AI Answer Bubbles             │
│   ├── Chunk Stats                  ├── Confidence Badges             │
│   ├── Indexed Files List           ├── Source Citations Panel        │
│   └── Clear & Start Over           └── Chat Input                    │
└──────────────┬───────────────────────────────┬──────────────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────────────────┐
│   INGESTION PIPELINE     │    │           Q&A PIPELINE               │
│     ingestion.py         │    │                                      │
│                          │    │  1. User submits question            │
│  ┌─────────────────────┐ │    │  2. Question embedded via HF API     │
│  │ PDF Files           │ │    │  3. FAISS similarity search (top-6)  │
│  │ └─ PyMuPDF          │ │    │  4. Normalize L2 scores → 0-1        │
│  │    Multi-mode       │ │    │  5. Compute Top-2 avg confidence     │
│  │    extraction       │ │    │  6. Route:                           │
│  │    (blocks/text/    │ │    │     High/Medium → Gemini LLM         │
│  │     dict modes)     │ │    │     Low         → "I don't know"     │
│  └─────────────────────┘ │    │  7. Gemini generates grounded answer │
│                          │    │  8. Extract + deduplicate sources    │
│  ┌─────────────────────┐ │    │  9. Return answer + citations        │
│  │ Image Files         │ │    └──────────────────────────────────────┘
│  │ └─ Compress image   │ │
│  │ └─ Gemini Vision    │ │    ┌──────────────────────────────────────┐
│  │    describes        │ │    │       CONFIDENCE SCORING             │
│  │    content          │ │    │                                      │
│  └─────────────────────┘ │    │  Raw FAISS L2 distance               │
│                          │    │       ↓                              │
│  ┌─────────────────────┐ │    │  Normalize: 1 / (1 + distance)       │
│  │ Text Cleaning       │ │    │       ↓                              │
│  │ └─ Remove noise     │ │    │  Top-2 Average Score                 │
│  │ └─ Fix ligatures    │ │    │       ↓                              │
│  │ └─ Normalize chars  │ │    │  ┌─────────────────────────────┐    │
│  └─────────────────────┘ │    │  │ Score ≥ 0.60 → 🟢 High     │    │
│                          │    │  │ Score ≥ 0.40 → 🟡 Medium   │    │
│  ┌─────────────────────┐ │    │  │ Score < 0.40 → 🔴 Low      │    │
│  │ Semantic Chunking   │ │    │  └─────────────────────────────┘    │
│  │ └─ HF Embeddings    │ │    └──────────────────────────────────────┘
│  │    detect topic     │ │
│  │    boundaries       │ │    ┌──────────────────────────────────────┐
│  │ └─ Split at meaning │ │    │       VECTOR STORE                   │
│  │    shifts           │ │    │       retrieval.py                   │
│  │ └─ Recursive        │ │    │                                      │
│  │    fallback for     │ │    │  HuggingFace Inference API           │
│  │    oversized chunks │ │    │  sentence-transformers/              │
│  └─────────────────────┘ │    │  all-MiniLM-L6-v2                   │
│                          │    │       ↓                              │
│  Metadata attached:      │    │  384-dimensional vectors             │
│  ├── source (filename)   │    │       ↓                              │
│  ├── type (pdf/image)    │    │  FAISS Local Index                   │
│  ├── page_number         │    │  (Persisted to disk)                 │
│  ├── image_name          │    │       ↓                              │
│  └── chunk_id            │    │  vectorstore/faiss_index/            │
└──────────────┬───────────┘    │  ├── index.faiss                     │
               │                │  └── index.pkl                       │
               └────────────────┴──────────────────────────────────────┘
```

---

## 🔄 Complete Project Flow

```
STEP 1: USER UPLOADS FILES
    User selects PDFs and/or images via Streamlit sidebar
                    │
                    ▼
STEP 2: FILES SAVED TO DISK
    save_uploaded_files() → saves to uploads/
                    │
                    ▼
STEP 3: DOCUMENT INGESTION
    For each file:
    ├── PDF  → PyMuPDF multi-mode extraction
    │          → Text cleaning
    │          → Semantic chunking (HF embeddings detect topic boundaries)
    │          → Recursive fallback for oversized chunks
    │          → Metadata: {source, type, page_number, chunk_id}
    │
    └── Image → Compress to ≤3MB JPEG
               → Gemini 2.5 Flash Vision API
               → Faithful text description extracted
               → Semantic chunking applied to description
               → Metadata: {source, type, image_name, chunk_id}
                    │
                    ▼
STEP 4: EMBEDDING + VECTOR STORE
    All chunks → HuggingFace Inference API
               → sentence-transformers/all-MiniLM-L6-v2
               → 384-dim vectors
               → FAISS index built
               → Saved to vectorstore/faiss_index/ (persistent)
                    │
                    ▼
STEP 5: USER ASKS A QUESTION
    User types question in Streamlit chat input
                    │
                    ▼
STEP 6: SEMANTIC RETRIEVAL
    Question → HF Embeddings → query vector
    FAISS similarity_search_with_score(query, k=6)
    → Returns top-6 chunks + L2 distances
                    │
                    ▼
STEP 7: CONFIDENCE SCORING
    For each score:
        normalized = 1 / (1 + L2_distance)
    top2_avg = average of top 2 normalized scores
    ├── top2_avg ≥ 0.60 → High   → proceed to LLM
    ├── top2_avg ≥ 0.40 → Medium → proceed to LLM
    └── top2_avg < 0.40 → Low    → trigger fallback (no LLM call)
                    │
                    ▼
STEP 8: ANSWER GENERATION (if not fallback)
    Retrieved chunks → format_context()
    → Source labels attached: [Source N: file.pdf | Page X]
    → Strict system prompt injected:
       "Answer ONLY from context. Always cite sources.
        Never fabricate. If absent, say I don't know."
    → Gemini 2.5 Flash generates answer
    → is_fallback_response() checks for genuine "I don't know"
                    │
                    ▼
STEP 9: RESPONSE DISPLAYED
    ├── Answer text (in chat bubble)
    ├── Confidence badge (🟢 High / 🟡 Medium / 🔴 Low)
    ├── Sources panel (filename + page number / image)
    └── Fallback badge if triggered
```
=======
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

### Core RAG Features
- 📄 **PDF Ingestion** — Multi-mode extraction (blocks, text, dict) handles columns, tables, complex layouts
- 🖼️ **Image Understanding** — Gemini 2.5 Flash Vision reads text, charts, diagrams, and tables from images
- 🧠 **Semantic Chunking** — HuggingFace embeddings detect topic boundaries for intelligent splitting
- 🔍 **FAISS Vector Search** — Local, persistent vector store with fast similarity search
- 📊 **Confidence Scoring** — Every answer scored High / Medium / Low based on retrieval similarity
- 🚫 **Hallucination Prevention** — Strict prompt engineering + Low confidence fallback
- 📚 **Source Citations** — Every claim cited with exact filename and page number
- 💾 **Persistent Index** — Vector store saved to disk, no re-embedding on restart
- 🔄 **Multi-file Support** — Query across multiple PDFs and images simultaneously

### UI Features
- 🌑 **Dark Theme** — GitHub-inspired dark color palette
- 💬 **Chat Interface** — Streamlit native chat with styled bubbles
- 🎨 **Color-coded Badges** — Green / Yellow / Red confidence indicators
- 📂 **File Status Panel** — Shows indexed files, chunk counts, system status
- 📈 **Chunk Statistics** — Total / PDF / Image chunk counts displayed
- 🗑️ **Reset Button** — Clear everything and start fresh

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.35 | Chat UI, file upload, display |
| **LLM** | Google Gemini 2.5 Flash | Answer generation |
| **Vision** | Google Gemini 2.5 Flash | Image content extraction |
| **Embeddings** | HuggingFace Inference API | Text vectorization |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim semantic vectors |
| **Semantic Chunking** | LangChain SemanticChunker | Topic-aware document splitting |
| **Vector Store** | FAISS (local) | Fast similarity search |
| **PDF Extraction** | PyMuPDF (fitz) | Multi-mode text extraction |
| **Image Processing** | Pillow | Compression before Vision API |
| **RAG Framework** | LangChain 0.2 | Pipeline orchestration |
| **Environment** | python-dotenv | Secure API key management |

---

## 📁 Project Structure

```
rag-document-qa/
│
├── app.py              # Streamlit frontend — dark UI, chat interface, sidebar
├── ingestion.py        # Document ingestion — PDF extraction + image vision + semantic chunking
├── retrieval.py        # Vector store — HF embeddings, FAISS, confidence scoring
├── generation.py       # Answer generation — Gemini LLM, prompt engineering, citations
├── config.py           # Central config — paths, models, thresholds, chunking params
│
├── uploads/            # Temporary uploaded files (git-ignored)
├── vectorstore/        # Persistent FAISS index (git-ignored)
│   └── faiss_index/
│       ├── index.faiss
│       └── index.pkl
│
├── utils/
│   └── __init__.py
│
├── .env                # API keys (git-ignored — never commit)
├── .gitignore          # Git ignore rules
├── requirements.txt    # All Python dependencies
└── README.md           # This file
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- A Google AI Studio API key (Gemini)
- A HuggingFace API token
- Git installed

### 1. Clone the Repository

```bash
git clone https://github.com/Aliraza-Amjad-Shaikh/rag-document-qa.git
cd rag-document-qa
```

### 2. Create Virtual Environment

```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Your API Keys

**Google Gemini API Key (Free):**
1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click **Create API Key**
3. Copy the key

**HuggingFace API Token (Free):**
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New Token**
3. Select **Read** access
4. Copy the token

### 5. Configure Environment

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your-gemini-api-key-here
HUGGINGFACEHUB_API_TOKEN=your-huggingface-token-here
```


### 6. Run the Application

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## 💡 How to Use

### Step 1: Upload Documents
- Click the file uploader in the sidebar
- Select one or more **PDF files** and/or **images** (JPG, PNG, WEBP)
- Click **🚀 Process & Index Documents**
- Wait for indexing to complete (progress bar shown)

### Step 2: Ask Questions
- Type your question in the chat input at the bottom
- Press **Enter** to submit

### Step 3: Read the Answer
- Answer appears with full source citations
- Confidence badge shows 🟢 High / 🟡 Medium / 🔴 Low
- Sources panel shows exact filename and page number

### Step 4: Reset When Done
- Click **🗑️ Clear & Start Over** in the sidebar to upload new documents

---

## ⚙️ Configuration Reference

All settings are centralized in `config.py`:

### Model Settings

| Setting | Value | Description |
|---|---|---|
| `VISION_MODEL` | `gemini-2.5-flash` | Gemini model for image understanding |
| `CHAT_MODEL` | `gemini-2.5-flash` | Gemini model for answer generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF model for vector embeddings |

### Chunking Settings

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Max characters for recursive fallback splits |
| `CHUNK_OVERLAP` | `150` | Overlap between fallback chunks |
| `SEMANTIC_BREAKPOINT_TYPE` | `percentile` | Strategy for detecting topic boundaries |
| `SEMANTIC_BREAKPOINT_THRESHOLD` | `85` | Sensitivity of semantic splits (higher = fewer, larger chunks) |
| `MAX_SEMANTIC_CHUNK_SIZE` | `1500` | Hard cap before recursive fallback triggers |
| `MIN_SEMANTIC_CHUNK_SIZE` | `50` | Minimum size — smaller chunks are dropped |

### Retrieval Settings

| Setting | Default | Description |
|---|---|---|
| `TOP_K_RESULTS` | `6` | Number of chunks retrieved per query |
| `CONFIDENCE_HIGH` | `0.60` | Threshold for High confidence |
| `CONFIDENCE_MEDIUM` | `0.40` | Threshold for Medium confidence |

---

## 🔬 How Semantic Chunking Works

Traditional character-based chunking splits text every N characters regardless of meaning:

```
❌ Character Chunking:
"Machine learning is a subset of AI. It learns from da" ← cut mid-word
"ta automatically without being explicitly programmed."
```

Semantic chunking uses embeddings to detect where the **meaning changes**:

```
✅ Semantic Chunking:
Chunk 1: "Machine learning is a subset of AI that learns
          from data automatically."           ← complete thought

Chunk 2: "Supervised learning uses labeled input-output
          pairs to train a model."            ← new topic, new chunk
```

**Process:**
1. Every sentence is embedded into a vector
2. Cosine similarity measured between consecutive sentences
3. Large similarity drops = topic boundary = split point
4. Oversized chunks are split further with recursive character splitting
5. Undersized chunks (< 50 chars) are dropped as noise

---

## 🔬 How Confidence Scoring Works

```
FAISS returns L2 distance (lower = more similar)
              ↓
Normalize: similarity = 1 / (1 + L2_distance)
              ↓
Take average of top-2 scores (more stable than best-only)
              ↓
┌────────────────────────────────────────────┐
│ top2_avg ≥ 0.60 → 🟢 High Confidence      │
│                   → LLM generates answer   │
│                                            │
│ top2_avg ≥ 0.40 → 🟡 Medium Confidence    │
│                   → LLM generates answer   │
│                                            │
│ top2_avg < 0.40 → 🔴 Low Confidence       │
│                   → Fallback triggered     │
│                   → LLM NOT called         │
│                   → Zero API cost          │
└────────────────────────────────────────────┘
```

---

## ⚠️ Limitations

| Limitation | Details |
|---|---|
| **Scanned PDFs** | PDFs that are scanned images cannot be processed by PyMuPDF. Use image upload instead. |
| **Large Documents** | 100+ page PDFs take significant time to embed. Semantic chunking adds extra HF API calls. |
| **No Conversation Memory** | Each question answered independently — no multi-turn context. |
| **No Document Summary** | Cannot summarize an entire document — answers are chunk-level only. |
| **Image Size Limit** | Images compressed to ≤3MB before Vision API. Very small or blurry images may produce poor descriptions. |
| **English Primary** | Best performance on English documents. Other languages may work but are untested. |
| **API Dependency** | Requires active Google and HuggingFace API keys. Offline use not supported. |
| **HF Rate Limits** | Free HuggingFace Inference API has rate limits. Large documents may hit these limits. |

---

## 🔮 Planned Improvements

- [ ] Conversation memory (multi-turn Q&A with context window)
- [ ] Document summarization (budget-aware map-reduce)
- [ ] DOCX / PPTX / CSV file format support
- [ ] Scanned PDF support via Tesseract OCR
- [ ] Chat history export (PDF / TXT)
- [ ] Document preview in sidebar
- [ ] Multi-session persistence
- [ ] Streamlit Cloud deployment
- [ ] Docker containerization
- [ ] HyDE (Hypothetical Document Embeddings) for better retrieval

---

## 📄 License

```
MIT License

Copyright (c) 2024 Aliraza Amjad Shaikh

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
```

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) — RAG orchestration and semantic chunking
- [Google Gemini](https://ai.google.dev/) — LLM and Vision capabilities
- [HuggingFace](https://huggingface.co/) — Inference API and embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF text extraction
- [Streamlit](https://streamlit.io) — Frontend framework
- [Sentence Transformers](https://www.sbert.net/) — all-MiniLM-L6-v2 embedding model

---

<p align="center">
  Built with ❤️ using LangChain · Google Gemini · HuggingFace · FAISS · Streamlit
</p>
<p align="center">
  <a href="https://github.com/Aliraza-Amjad-Shaikh/rag-document-qa">
    ⭐ Star this repo if you found it useful!
  </a>
</p>
```

---
