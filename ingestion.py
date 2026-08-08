import os
import re
import fitz  # PyMuPDF
import base64
from typing import List
from PIL import Image
import io

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    # OPENAI_API_KEY,
    VISION_MODEL,
    EMBEDDING_MODEL,
    SEMANTIC_BREAKPOINT_TYPE,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    MAX_SEMANTIC_CHUNK_SIZE,
    MIN_SEMANTIC_CHUNK_SIZE,
    CHAT_MODEL
)

from document_store import add_document_entry

# ─────────────────────────────────────────────
# Initialize Semantic Chunker
# ─────────────────────────────────────────────

def get_semantic_chunker(api_key: str) -> SemanticChunker:
    """
    Initialize LangChain SemanticChunker using OpenAI
    embeddings to detect topic boundaries in text.

    The chunker embeds every sentence, measures how much
    the meaning changes between consecutive sentences,
    and splits at the points of highest semantic shift.

    Returns:
        SemanticChunker instance
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key
    )

    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD
    )

def get_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """
    Fallback splitter for oversized semantic chunks.
    Splits large chunks further while preserving sentence boundaries.

    Returns:
        RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )

# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted text — remove noise, normalize whitespace.

    Args:
        text: Raw extracted text string

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize unicode ligatures and special characters
    replacements = {
        '\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff',
        '\ufb03': 'ffi', '\ufb04': 'ffl', '\u2019': "'",
        '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--', '\u2022': '*',
        '\u00a0': ' ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Clean whitespace per line
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = re.sub(r'[ \t]+', ' ', line).strip()
        cleaned.append(line)

    # Remove excessive blank lines
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Filter noise-only lines
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Skip standalone page numbers
        if re.match(r'^\d{1,3}$', stripped):
            continue
        # Skip lines with only special characters
        if stripped and not re.match(r'^[^a-zA-Z0-9]+$', stripped):
            filtered.append(line)
        elif not stripped:
            filtered.append(line)

    return '\n'.join(filtered).strip()

# ─────────────────────────────────────────────
# PDF Text Extraction (Multi-Mode)
# ─────────────────────────────────────────────

def extract_text_from_page(page: fitz.Page) -> str:
    """
    Extract text from a PDF page using 3 strategies
    in order of preference.

    Strategy 1: Blocks mode — best for columns/tables
    Strategy 2: Raw text — simple fallback
    Strategy 3: Dict mode — deep fallback for complex layouts

    Args:
        page: PyMuPDF Page object

    Returns:
        Cleaned text string
    """
    full_text = ""

    # Strategy 1: Blocks
    try:
        blocks = page.get_text("blocks", sort=True)
        block_texts = []
        for block in blocks:
            if len(block) >= 5:
                block_type = block[6] if len(block) > 6 else 0
                if block_type == 0:
                    t = block[4].strip()
                    if t:
                        block_texts.append(t)
        full_text = "\n\n".join(block_texts)
    except Exception:
        pass

    # Strategy 2: Raw text fallback
    if not full_text.strip():
        try:
            full_text = page.get_text("text")
        except Exception:
            pass

    # Strategy 3: Dict mode fallback
    if not full_text.strip():
        try:
            page_dict = page.get_text("dict")
            dict_texts = []
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        line_text = " ".join(
                            span.get("text", "")
                            for span in line.get("spans", [])
                        ).strip()
                        if line_text:
                            dict_texts.append(line_text)
            full_text = "\n".join(dict_texts)
        except Exception:
            pass

    return clean_text(full_text)

def extract_pdf_pages(pdf_path: str) -> List[dict]:
    """
    Extract text from every page of a PDF.

    Args:
        pdf_path: Absolute path to PDF file

    Returns:
        List of dicts: {text, page_number, source}
    """
    pages = []
    filename = os.path.basename(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        print(f"[PDF] Found {len(doc)} pages in '{filename}'")

        for page_index in range(len(doc)):
            page = doc[page_index]
            text = extract_text_from_page(page)

            if not text.strip():
                print(f"  [SKIP] Page {page_index + 1} — no extractable text")
                continue

            pages.append({
                "text": text,
                "page_number": page_index + 1,
                "source": filename
            })
            print(f"  [PAGE {page_index + 1}] Extracted {len(text)} chars")

        doc.close()

    except Exception as e:
        print(f"[ERROR] PDF extraction failed for '{filename}': {e}")

    return pages

# ─────────────────────────────────────────────
# Semantic Chunking Pipeline
# ─────────────────────────────────────────────

def semantic_chunk_text(
    text: str,
    source: str,
    page_number: int,
    doc_type: str,
    api_key: str,
    chunk_id_start: int = 0
) -> List[Document]:
    """
    Apply semantic chunking to a block of text.

    Process:
    1. SemanticChunker splits at topic boundaries
    2. Oversized chunks are split further with RecursiveCharacterTextSplitter
    3. Undersized chunks are dropped
    4. Every chunk gets full metadata attached

    Args:
        text: Raw text to chunk
        source: Filename for metadata
        page_number: Page number for metadata (0 for images)
        doc_type: 'pdf' or 'image'
        chunk_id_start: Starting chunk ID for this batch

    Returns:
        List of LangChain Document objects
    """
    if not text.strip():
        return []

    semantic_chunker = get_semantic_chunker(api_key)
    recursive_splitter = get_recursive_splitter()

    # ── Step 1: Semantic split ──
    try:
        raw_semantic_chunks = semantic_chunker.split_text(text)
        print(f"  [SEMANTIC] '{source}' p{page_number} → "
              f"{len(raw_semantic_chunks)} semantic chunks")
    except Exception as e:
        print(f"  [SEMANTIC] Fallback to recursive — semantic failed: {e}")
        raw_semantic_chunks = recursive_splitter.split_text(text)

    # ── Step 2: Process each semantic chunk ──
    final_chunks = []
    chunk_id = chunk_id_start

    for sem_chunk in raw_semantic_chunks:
        sem_chunk = sem_chunk.strip()

        # Drop noise chunks
        if len(sem_chunk) < MIN_SEMANTIC_CHUNK_SIZE:
            continue

        # If chunk is within max size → keep as is
        if len(sem_chunk) <= MAX_SEMANTIC_CHUNK_SIZE:
            metadata = {
                "source": source,
                "type": doc_type,
                "chunk_id": chunk_id
            }
            if doc_type == "pdf":
                metadata["page_number"] = page_number
            elif doc_type == "image":
                metadata["image_name"] = source

            final_chunks.append(Document(
                page_content=sem_chunk,
                metadata=metadata
            ))
            chunk_id += 1

        # If chunk exceeds max size → split further
        else:
            print(f"  [SEMANTIC] Oversized chunk ({len(sem_chunk)} chars) "
                  f"→ splitting further")
            sub_chunks = recursive_splitter.split_text(sem_chunk)

            for sub in sub_chunks:
                sub = sub.strip()
                if len(sub) < MIN_SEMANTIC_CHUNK_SIZE:
                    continue

                metadata = {
                    "source": source,
                    "type": doc_type,
                    "chunk_id": chunk_id
                }
                if doc_type == "pdf":
                    metadata["page_number"] = page_number
                elif doc_type == "image":
                    metadata["image_name"] = source

                final_chunks.append(Document(
                    page_content=sub,
                    metadata=metadata
                ))
                chunk_id += 1

    return final_chunks

# ─────────────────────────────────────────────
# Document Summary Generation
# ─────────────────────────────────────────────

def generate_document_summary(full_text: str, filename: str, api_key: str) -> str:
    """
    Generate a short summary of a document's full text using the LLM.
    Called once per file at ingestion time — not per question.

    Note: only the first 12,000 characters are sent to the LLM.
    Very large documents will be summarized based on that initial
    portion only, not the entire text.

    Args:
        full_text: Combined raw text of the document (all pages/description)
        filename: Name of the file, used for context in the prompt

    Returns:
        Summary string (4-6 sentences)
    """
    try:
        llm = ChatOpenAI(model=CHAT_MODEL, api_key=api_key, temperature=0)
        prompt = f"""Summarize this document in 4-6 sentences, covering its main topic, purpose, and key sections.

Document: {filename}

Content:
{full_text[:12000]}

Summary:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        print(f"  [SUMMARY] ✅ Generated summary for '{filename}' ({len(summary)} chars)")
        return summary
    except Exception as e:
        print(f"  [SUMMARY] ❌ Failed to generate summary for '{filename}': {e}")
        return "Summary could not be generated for this document."

# ─────────────────────────────────────────────
# PDF Ingestion Pipeline
# ─────────────────────────────────────────────

def ingest_pdf(pdf_path: str, api_key: str) -> List[Document]:
    """
    Full PDF ingestion pipeline:
    extract text (multi-mode) → clean → semantic chunk → metadata → summary

    Args:
        pdf_path: Absolute path to PDF file

    Returns:
        List of semantically chunked Document objects
    """
    print(f"\n[PDF] Ingesting: {os.path.basename(pdf_path)}")
    pages = extract_pdf_pages(pdf_path)
    all_chunks = []
    chunk_id = 0

    for page in pages:
        chunks = semantic_chunk_text(
            text=page["text"],
            source=page["source"],
            page_number=page["page_number"],
            doc_type="pdf",
            api_key=api_key,
            chunk_id_start=chunk_id
        )
        all_chunks.extend(chunks)
        chunk_id += len(chunks)

    print(f"[PDF] ✅ {len(pages)} pages → {len(all_chunks)} semantic chunks")

    # ── Generate summary + save metadata entry ──
    filename = os.path.basename(pdf_path)
    full_text = "\n\n".join(p["text"] for p in pages)
    summary = generate_document_summary(full_text, filename)
    add_document_entry(
        filename=filename,
        doc_type="pdf",
        page_count=len(pages),
        chunk_count=len(all_chunks),
        summary=summary
    )

    return all_chunks

# ─────────────────────────────────────────────
# Image Compression
# ─────────────────────────────────────────────

def compress_image(image_path: str, max_size_mb: float = 3.0) -> bytes:
    """
    Compress and resize image to stay within API size limits.

    Args:
        image_path: Path to image file
        max_size_mb: Maximum size in MB before compression

    Returns:
        Compressed image as bytes (JPEG format)
    """
    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Resize if too large
        max_dim = 2048
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            print(f"  [IMAGE] Resized to {new_size[0]}x{new_size[1]}")

        # Compress iteratively
        buffer = io.BytesIO()
        quality = 85
        while True:
            buffer.seek(0)
            buffer.truncate()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            size_mb = buffer.tell() / (1024 * 1024)
            if size_mb <= max_size_mb or quality <= 30:
                print(f"  [IMAGE] Compressed to {size_mb:.2f} MB (quality={quality})")
                break
            quality -= 10

        buffer.seek(0)
        return buffer.read()

# ─────────────────────────────────────────────
# Image Ingestion (OpenAI Vision)
# ─────────────────────────────────────────────

def describe_image_with_vision(image_path: str, api_key: str) -> str:
    """
    Use OpenAI Vision (gpt-4o-mini) to extract and describe image content.

    Args:
        image_path: Absolute path to image

    Returns:
        Faithful text description of image content
    """
    filename = os.path.basename(image_path)
    print(f"  [IMAGE] Sending to OpenAI Vision: {filename}")

    prompt = """You are a precise document analysis assistant.
Analyze this image thoroughly and extract ALL of the following:

1. ALL visible text — extract it word for word exactly as it appears
2. Tables — describe every row and column with their exact values
3. Charts/Graphs — type, axes, labels, and all key data points
4. Diagrams — all components, labels, and relationships
5. Lists or bullet points — extract them completely
6. Headers, titles, captions — extract them all
7. Numbers, dates, statistics — extract them precisely

IMPORTANT RULES:
- Be exhaustive — do not skip any visible text or data
- Be faithful — do not add information not in the image
- Do NOT say 'I can see...' — directly state the content
- Format as clean, well-structured text
- Mark partially visible text as [partial: ...]"""

    try:
        image_bytes = compress_image(image_path)
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        llm = ChatOpenAI(
            model=VISION_MODEL,
            api_key=api_key,
            temperature=0
        )

        response = llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                }
            ])
        ])

        description = response.content.strip()
        print(f"  [IMAGE] ✅ OpenAI extracted {len(description)} characters")
        return description

    except Exception as e:
        print(f"  [IMAGE] ❌ OpenAI Vision failed for '{filename}': {e}")
        return f"Image content could not be extracted: {e}"

def ingest_image(image_path: str, api_key: str) -> List[Document]:
    """
    Full image ingestion pipeline:
    compress → OpenAI Vision describes → semantic chunk → metadata → summary

    Args:
        image_path: Absolute path to image

    Returns:
        List of Document chunks representing the image
    """
    filename = os.path.basename(image_path)
    print(f"\n[IMAGE] Ingesting: {filename}")
    description = describe_image_with_vision(image_path, api_key)

    if not description.strip():
        print(f"[IMAGE] ❌ Empty description for {filename}")
        return []

    # Apply semantic chunking to image description
    chunks = semantic_chunk_text(
        text=description,
        source=filename,
        page_number=0,
        doc_type="image",
        api_key=api_key,
        chunk_id_start=0
    )

    # If semantic chunking produced nothing, wrap as single chunk
    if not chunks:
        chunks = [Document(
            page_content=description,
            metadata={
                "source": filename,
                "type": "image",
                "image_name": filename,
                "chunk_id": 0
            }
        )]

    print(f"[IMAGE] ✅ {len(chunks)} semantic chunk(s) produced")

    # ── Generate summary + save metadata entry ──
    summary = generate_document_summary(description, filename, api_key)
    add_document_entry(
        filename=filename,
        doc_type="image",
        page_count=None,
        chunk_count=len(chunks),
        summary=summary
    )

    return chunks

# ─────────────────────────────────────────────
# Save Uploaded Files (Streamlit)
# ─────────────────────────────────────────────

def save_uploaded_files(uploaded_files) -> List[str]:
    """
    Save Streamlit uploaded files to the uploads directory.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        List of saved file paths
    """
    saved_paths = []
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file_path)
        print(f"[SAVE] Saved: {file.name}")

    return saved_paths

# ─────────────────────────────────────────────
# Unified Ingestion Pipeline
# ─────────────────────────────────────────────

def ingest_documents(file_paths: List[str], api_key: str) -> List[Document]:
    """
    Main entry point — routes files to correct pipeline.

    Args:
        file_paths: List of absolute file paths

    Returns:
        Combined list of all semantic Document chunks
    """
    all_documents = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"[SKIP] Unsupported: {filename}")
            continue

        try:
            if ext in PDF_EXTENSIONS:
                docs = ingest_pdf(path, api_key)
            elif ext in IMAGE_EXTENSIONS:
                docs = ingest_image(path, api_key)
            else:
                continue
            all_documents.extend(docs)

        except Exception as e:
            print(f"[ERROR] Failed processing '{filename}': {e}")

    print(f"\n✅ TOTAL SEMANTIC CHUNKS READY: {len(all_documents)}")
    return all_documents

# ─────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <file1> <file2> ...")
        sys.exit(1)

    file_paths = [p for p in sys.argv[1:] if os.path.exists(p)]
    documents = ingest_documents(file_paths)

    print("\n" + "="*60)
    print(f"TOTAL SEMANTIC CHUNKS: {len(documents)}")
    print("="*60)

    for i, doc in enumerate(documents[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source   : {doc.metadata.get('source')}")
        print(f"Type     : {doc.metadata.get('type')}")
        print(f"Page     : {doc.metadata.get('page_number', 'N/A')}")
        print(f"Chunk ID : {doc.metadata.get('chunk_id')}")
        print(f"Length   : {len(doc.page_content)} chars")
        print(f"Preview  : {doc.page_content[:200]}...")
        print(f"{'─'*60}")