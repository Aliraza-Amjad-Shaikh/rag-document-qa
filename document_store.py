import json
import os
from datetime import datetime
from config import BASE_DIR

DOCUMENT_STORE_PATH = os.path.join(BASE_DIR, "vectorstore", "document_store.json")

def load_document_store() -> dict:
    if not os.path.exists(DOCUMENT_STORE_PATH):
        return {}
    with open(DOCUMENT_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_document_store(store: dict) -> None:
    os.makedirs(os.path.dirname(DOCUMENT_STORE_PATH), exist_ok=True)
    with open(DOCUMENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)

def add_document_entry(filename, doc_type, page_count, chunk_count, summary):
    store = load_document_store()
    store[filename] = {
        "type": doc_type,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "summary": summary,
        "uploaded_at": datetime.now().isoformat()
    }
    save_document_store(store)

def remove_document_entry(filename):
    store = load_document_store()
    if filename in store:
        del store[filename]
        save_document_store(store)

def get_all_documents_summary_text() -> str:
    store = load_document_store()
    parts = [
        f"File: {fname} ({meta['type']}, {meta.get('page_count', 'N/A')} pages)\nSummary: {meta['summary']}"
        for fname, meta in store.items()
    ]
    return "\n\n".join(parts)