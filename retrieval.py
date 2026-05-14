import os
import shutil
from typing import List
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    HUGGINGFACEHUB_API_TOKEN,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    FAISS_INDEX_NAME,
    TOP_K_RESULTS,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM
)


# ─────────────────────────────────────────────
# Embeddings Initialization
# ─────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """
    Initialize HuggingFace Inference API embeddings.
    Uses sentence-transformers/all-MiniLM-L6-v2 model.
    """
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )


# ─────────────────────────────────────────────
# Build Vector Store
# ─────────────────────────────────────────────

def build_vectorstore(documents: List[Document]) -> FAISS:
    """
    Embed all chunks and store in FAISS.
    """
    if not documents:
        raise ValueError("[ERROR] No documents provided.")

    embeddings  = get_embeddings()
    print(f"[VECTORSTORE] Embedding {len(documents)} chunks using {EMBEDDING_MODEL}...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("[VECTORSTORE] ✅ Embedding complete.")
    return vectorstore


# ─────────────────────────────────────────────
# Save Vector Store
# ─────────────────────────────────────────────

def save_vectorstore(vectorstore: FAISS) -> None:
    """Save FAISS index to disk."""
    save_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)
    vectorstore.save_local(save_path)
    print(f"[VECTORSTORE] ✅ Saved to: {save_path}")


# ─────────────────────────────────────────────
# Load Vector Store
# ─────────────────────────────────────────────

def load_vectorstore() -> FAISS | None:
    """Load existing FAISS index from disk."""
    save_path  = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)
    index_file = os.path.join(save_path, "index.faiss")

    if not os.path.exists(index_file):
        print("[VECTORSTORE] No existing index found.")
        return None

    embeddings  = get_embeddings()
    print("[VECTORSTORE] Loading existing index...")
    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("[VECTORSTORE] ✅ Index loaded.")
    return vectorstore


# ─────────────────────────────────────────────
# Confidence Scoring
# ─────────────────────────────────────────────

def normalize_score(raw_score: float) -> float:
    """Convert FAISS L2 distance to 0-1 similarity."""
    return 1 / (1 + raw_score)


def compute_confidence(score: float) -> str:
    """Convert similarity score to confidence label."""
    if score >= CONFIDENCE_HIGH:
        return "High"
    elif score >= CONFIDENCE_MEDIUM:
        return "Medium"
    else:
        return "Low"


# ─────────────────────────────────────────────
# Retrieval Pipeline
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(query: str, vectorstore: FAISS) -> dict:
    """
    Retrieve top-k relevant chunks with confidence scoring.
    """
    results = vectorstore.similarity_search_with_score(query, k=TOP_K_RESULTS)

    if not results:
        return {
            "chunks":        [],
            "scores":        [],
            "confidence":    "Low",
            "should_answer": False
        }

    chunks = []
    scores = []

    for doc, raw_score in results:
        normalized = normalize_score(raw_score)
        chunks.append(doc)
        scores.append(normalized)

    best_score    = max(scores)
    top2_avg      = sum(sorted(scores, reverse=True)[:2]) / 2
    confidence    = compute_confidence(top2_avg)
    should_answer = confidence != "Low"

    print(f"\n[RETRIEVAL] Query: '{query}'")
    print(f"[RETRIEVAL] Best: {best_score:.4f} | Top2 avg: {top2_avg:.4f} → {confidence}")
    print(f"[RETRIEVAL] Should answer: {should_answer}")
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        print(f"  Chunk {i+1}: score={score:.4f} | "
              f"source={chunk.metadata.get('source')} | "
              f"page={chunk.metadata.get('page_number', 'N/A')}")

    return {
        "chunks":        chunks,
        "scores":        scores,
        "confidence":    confidence,
        "should_answer": should_answer
    }


# ─────────────────────────────────────────────
# Master Vector Store Handler
# ─────────────────────────────────────────────

def get_or_build_vectorstore(documents: List[Document] = None) -> FAISS | None:
    """
    If documents provided → always rebuild fresh.
    If not → try loading existing index.
    """
    if documents:
        print(f"[VECTORSTORE] Building fresh index from {len(documents)} chunks...")
        save_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
            print("[VECTORSTORE] Cleared old index.")
        vectorstore = build_vectorstore(documents)
        save_vectorstore(vectorstore)
        return vectorstore

    vectorstore = load_vectorstore()
    if vectorstore:
        print("[VECTORSTORE] Using existing index.")
        return vectorstore

    print("[VECTORSTORE] No index found and no documents provided.")
    return None


# ─────────────────────────────────────────────
# Clear Vector Store
# ─────────────────────────────────────────────

def clear_vectorstore() -> None:
    """Delete FAISS index from disk."""
    save_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print("[VECTORSTORE] ✅ Index cleared.")
    else:
        print("[VECTORSTORE] Nothing to clear.")


# ─────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion import ingest_documents
    import sys

    if len(sys.argv) < 2:
        print("Usage: python retrieval.py <file1> <file2> ...")
        sys.exit(1)

    file_paths  = sys.argv[1:]
    existing_vs = load_vectorstore()

    if existing_vs:
        vs = existing_vs
    else:
        docs = ingest_documents(file_paths)
        vs   = build_vectorstore(docs)
        save_vectorstore(vs)

    test_queries = [
        "What is Machine Learning?",
        "What is the weather like today?",
    ]

    for query in test_queries:
        print(f"\n{'─'*50}")
        result = retrieve_relevant_chunks(query, vs)
        print(f"Confidence   : {result['confidence']}")
        print(f"Should Answer: {result['should_answer']}")