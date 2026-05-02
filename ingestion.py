import os
import re
import fitz  # PyMuPDF
import base64
from typing import List
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    OPENAI_API_KEY,
    VISION_MODEL
)

# ─────────────────────────────────────────────
# Text Cleaning Utilities
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing noise while
    preserving meaningful content and structure.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Remove null bytes and control characters (except newlines/tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize unicode ligatures and special chars
    replacements = {
        '\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff',
        '\ufb03': 'ffi', '\ufb04': 'ffl', '\u2019': "'",
        '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--', '\u2022': '*',
        '\u00a0': ' ',  # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove excessive whitespace within lines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'[ \t]+', ' ', line).strip()
        cleaned_lines.append(line)

    # Remove excessive blank lines (keep max 2 consecutive)
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just noise (single chars, page numbers alone)
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just a number (page numbers)
        if re.match(r'^\d{1,3}$', stripped):
            continue
        # Skip lines with only special characters
        if stripped and not re.match(r'^[^a-zA-Z0-9]+$', stripped):
            filtered.append(line)
        elif not stripped:
            filtered.append(line)

    return '\n'.join(filtered).strip()


# ─────────────────────────────────────────────
# PDF Ingestion (Enhanced Multi-Mode Extraction)
# ─────────────────────────────────────────────

def extract_text_from_page(page: fitz.Page) -> str:
    """
    Extract text from a single PDF page using multiple
    extraction strategies for maximum coverage.

    Strategy:
    1. Try 'blocks' mode — best for multi-column and tables
    2. Fallback to standard text extraction
    3. Clean and merge results

    Args:
        page: PyMuPDF Page object

    Returns:
        Cleaned text string for the page
    """
    full_text = ""

    try:
        # Strategy 1: Extract as structured blocks (handles columns/tables)
        blocks = page.get_text("blocks", sort=True)
        block_texts = []

        for block in blocks:
            # block format: (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) >= 5:
                block_type = block[6] if len(block) > 6 else 0
                # block_type 0 = text, 1 = image
                if block_type == 0:
                    text = block[4].strip()
                    if text:
                        block_texts.append(text)

        full_text = "\n\n".join(block_texts)

    except Exception:
        pass

    # Strategy 2: Fallback to raw text if blocks gave nothing
    if not full_text.strip():
        try:
            full_text = page.get_text("text")
        except Exception:
            pass

    # Strategy 3: Try dict mode for any remaining gaps
    if not full_text.strip():
        try:
            page_dict = page.get_text("dict")
            dict_texts = []
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # text block
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
    Extract text from every page of a PDF using enhanced
    multi-mode extraction.

    Args:
        pdf_path: Absolute path to the PDF file

    Returns:
        List of dicts: {text, page_number, source}
    """
    pages     = []
    filename  = os.path.basename(pdf_path)

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
                "text":        text,
                "page_number": page_index + 1,
                "source":      filename
            })
            print(f"  [PAGE {page_index + 1}] Extracted {len(text)} characters")

        doc.close()

    except Exception as e:
        print(f"[ERROR] Failed to extract PDF '{filename}': {e}")

    return pages


def chunk_pdf_text(pages: List[dict]) -> List[Document]:
    """
    Split extracted PDF pages into overlapping chunks.
    Uses smart separators to avoid splitting mid-sentence.

    Args:
        pages: Output from extract_pdf_pages()

    Returns:
        List of LangChain Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )

    all_chunks = []
    chunk_id   = 0

    for page in pages:
        raw_chunks = splitter.split_text(page["text"])

        for chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            # Skip very short chunks (likely noise)
            if len(chunk_text) < 30:
                continue

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source":      page["source"],
                    "type":        "pdf",
                    "page_number": page["page_number"],
                    "chunk_id":    chunk_id
                }
            )
            all_chunks.append(doc)
            chunk_id += 1

    return all_chunks


def ingest_pdf(pdf_path: str) -> List[Document]:
    """
    Full PDF ingestion pipeline:
    extract (multi-mode) → clean → chunk → attach metadata

    Args:
        pdf_path: Absolute path to the PDF

    Returns:
        List of LangChain Document chunks
    """
    print(f"\n[PDF] Ingesting: {os.path.basename(pdf_path)}")
    pages  = extract_pdf_pages(pdf_path)
    chunks = chunk_pdf_text(pages)
    print(f"[PDF] ✅ {len(pages)} pages → {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# Image Ingestion (GPT-4o Vision)
# ─────────────────────────────────────────────

def encode_image_to_base64(image_path: str) -> str:
    """
    Convert image to base64 for OpenAI Vision API.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image_with_vision(image_path: str) -> str:
    """
    Use GPT-4o Vision to extract and describe image content.

    Args:
        image_path: Absolute path to image

    Returns:
        Detailed text description of image content
    """
    client   = OpenAI(api_key=OPENAI_API_KEY)
    filename = os.path.basename(image_path)
    ext      = os.path.splitext(filename)[1].lower()

    media_type_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp"
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    image_data = encode_image_to_base64(image_path)

    prompt = """You are a precise document analysis assistant.
Analyze this image thoroughly and extract ALL of the following:

1. ALL visible text — extract it word for word exactly as it appears
2. Tables — describe every row and column with their exact values
3. Charts/Graphs — describe type, axes, labels, and all data points
4. Diagrams — describe all components, labels, and their relationships
5. Lists or bullet points — extract them completely
6. Headers, titles, captions — extract them all
7. Any numbers, dates, statistics — extract them precisely

IMPORTANT RULES:
- Be exhaustive — do not skip any visible text or data
- Be faithful — do not add information not visible in the image
- Do NOT say 'I can see...' — directly state the content
- Format as clean, well-structured text
- If text is partially visible, mark it as [partial: ...]"""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":    f"data:{media_type};base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=3000
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] Vision API failed for '{filename}': {e}")
        return f"Image content could not be extracted: {e}"


def image_to_document(image_path: str) -> Document:
    """
    Convert image to a LangChain Document via GPT-4o Vision.

    Args:
        image_path: Absolute path to image

    Returns:
        LangChain Document with image description
    """
    filename    = os.path.basename(image_path)
    description = describe_image_with_vision(image_path)

    return Document(
        page_content=description,
        metadata={
            "source":     filename,
            "type":       "image",
            "image_name": filename,
            "chunk_id":   0
        }
    )


def ingest_image(image_path: str) -> List[Document]:
    """
    Full image ingestion pipeline.

    Args:
        image_path: Absolute path to image

    Returns:
        List containing Document(s) for the image
    """
    filename = os.path.basename(image_path)
    print(f"\n[IMAGE] Ingesting: {filename}")
    doc = image_to_document(image_path)

    # If image description is long, chunk it too
    if len(doc.page_content) > CHUNK_SIZE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks_text = splitter.split_text(doc.page_content)
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            if len(chunk_text.strip()) < 30:
                continue
            chunks.append(Document(
                page_content=chunk_text.strip(),
                metadata={
                    "source":     filename,
                    "type":       "image",
                    "image_name": filename,
                    "chunk_id":   i
                }
            ))
        print(f"[IMAGE] ✅ Description chunked into {len(chunks)} parts")
        return chunks

    print(f"[IMAGE] ✅ Extracted {len(doc.page_content)} characters")
    return [doc]


# ─────────────────────────────────────────────
# Save Uploaded Files (from Streamlit)
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

def ingest_documents(file_paths: List[str]) -> List[Document]:
    """
    Main ingestion pipeline — routes files to correct processor.

    Args:
        file_paths: List of absolute file paths

    Returns:
        Combined list of all Document chunks
    """
    all_documents = []

    for path in file_paths:
        ext      = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)

        if ext not in SUPPORTED_EXTENSIONS:
            print(f"[SKIP] Unsupported file type: {filename}")
            continue

        try:
            if ext in PDF_EXTENSIONS:
                docs = ingest_pdf(path)
            elif ext in IMAGE_EXTENSIONS:
                docs = ingest_image(path)
            else:
                continue

            all_documents.extend(docs)

        except Exception as e:
            print(f"[ERROR] Failed processing '{filename}': {e}")

    print(f"\n✅ TOTAL CHUNKS READY FOR EMBEDDING: {len(all_documents)}")
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
    documents  = ingest_documents(file_paths)

    print("\n" + "="*50)
    print(f"TOTAL CHUNKS: {len(documents)}")
    print("="*50)

    for i, doc in enumerate(documents[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source : {doc.metadata.get('source')}")
        print(f"Type   : {doc.metadata.get('type')}")
        print(f"Preview: {doc.page_content[:300]}...")