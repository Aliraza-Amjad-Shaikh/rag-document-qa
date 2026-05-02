import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import OPENAI_API_KEY, VECTORSTORE_DIR, FAISS_INDEX_NAME, EMBEDDING_MODEL

# ─────────────────────────────────────────────
# Embedding Model Initialization
# ─────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    """
    Initialize and return the OpenAI embedding model.
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )


# ─────────────────────────────────────────────
# Build Vector Store from Documents
# ─────────────────────────────────────────────

def build_vectorstore(documents: list[Document]) -> FAISS:
    """
    Embed all document chunks and create a FAISS vector store.

    Args:
        documents: List of LangChain Document objects from ingestion

    Returns:
        FAISS vector store instance
    """
    if not documents:
        raise ValueError("[ERROR] No documents provided to build vector store.")

    embeddings = get_embeddings()
    print(f"[VECTORSTORE] Embedding {len(documents)} chunks using {EMBEDDING_MODEL}...")
    
    # FAISS.from_documents handles batching and embedding automatically
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print("[VECTORSTORE] Embedding complete.")
    return vectorstore


# ─────────────────────────────────────────────
# Save Vector Store to Disk (Persistence)
# ─────────────────────────────────────────────

def save_vectorstore(vectorstore: FAISS) -> None:
    """
    Save the FAISS index and document store to disk.

    Args:
        vectorstore: FAISS instance to save
    """
    save_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)
    vectorstore.save_local(save_path)
    print(f"[VECTORSTORE] Saved index to: {save_path}")


# ─────────────────────────────────────────────
# Load Existing Vector Store from Disk
# ─────────────────────────────────────────────

def load_vectorstore() -> FAISS | None:
    """
    Load a previously saved FAISS index from disk.

    Returns:
        FAISS instance if found, else None
    """
    save_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)
    # NEW - Replace with this
    index_file = os.path.join(save_path, "index.faiss")

    if not os.path.exists(index_file):
        print("[VECTORSTORE] No existing index found. Will build new one.")
        return None

    embeddings = get_embeddings()
    print("[VECTORSTORE] Loading existing index from disk...")
    
    # allow_dangerous_deserialization=True is required in langchain-community >= 0.2.0
    vectorstore = FAISS.load_local(
        save_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print("[VECTORSTORE] Index loaded successfully.")
    return vectorstore

from config import TOP_K_RESULTS, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

# ─────────────────────────────────────────────
# Confidence Scoring
# ─────────────────────────────────────────────

def compute_confidence(score: float) -> str:
    """
    Convert a FAISS similarity score into a human-readable
    confidence label.

    Note: FAISS returns L2 distance scores (lower = more similar).
    We convert them to a 0-1 similarity scale first.

    Args:
        score: Raw FAISS similarity score

    Returns:
        "High", "Medium", or "Low"
    """
    if score >= CONFIDENCE_HIGH:
        return "High"
    elif score >= CONFIDENCE_MEDIUM:
        return "Medium"
    else:
        return "Low"


def normalize_score(raw_score: float) -> float:
    """
    Normalize a FAISS L2 distance score to a 0-1 similarity scale.
    FAISS L2: score=0 means identical, higher means more distant.
    We convert to similarity: 1 / (1 + distance)

    Args:
        raw_score: Raw L2 distance from FAISS

    Returns:
        Normalized similarity score between 0 and 1
    """
    return 1 / (1 + raw_score)


# ─────────────────────────────────────────────
# Retrieval Pipeline
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(query: str, vectorstore: FAISS) -> dict:
    """
    Retrieve relevant chunks and compute confidence score.
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

    best_score  = max(scores)
    # Use average of top-2 scores for more stable confidence
    top2_avg    = sum(sorted(scores, reverse=True)[:2]) / 2
    confidence  = compute_confidence(top2_avg)
    should_answer = confidence != "Low"

    print(f"\n[RETRIEVAL] Query: '{query}'")
    print(f"[RETRIEVAL] Best score: {best_score:.4f} | Top-2 avg: {top2_avg:.4f} → Confidence: {confidence}")
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
# (Used by Streamlit in Phase 6)
# ─────────────────────────────────────────────

def get_or_build_vectorstore(documents: list[Document] = None) -> FAISS | None:
    """
    Smart loader with merge capability.

    If documents are provided → always build fresh from those documents.
    If no documents provided → try loading existing index.
    If nothing available → return None.

    Args:
        documents: List of Document chunks from ingestion pipeline.
                   If provided, ALWAYS rebuilds from scratch.

    Returns:
        FAISS instance or None
    """
    # If new documents are provided, always rebuild fresh
    if documents:
        print(f"[VECTORSTORE] Building fresh index from {len(documents)} chunks...")

        # Clear old index first to avoid stale data
        save_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
            print("[VECTORSTORE] Cleared old index before rebuild.")

        vectorstore = build_vectorstore(documents)
        save_vectorstore(vectorstore)
        return vectorstore

    # No documents provided — try loading existing index
    vectorstore = load_vectorstore()
    if vectorstore:
        print("[VECTORSTORE] Using existing index.")
        return vectorstore

    # Nothing available
    print("[VECTORSTORE] No index found and no documents provided.")
    return None


def clear_vectorstore() -> None:
    """
    Delete the existing FAISS index from disk.
    Called when user clicks 'Clear and Upload New Documents' in Streamlit.
    """
    import shutil
    save_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print("[VECTORSTORE] Existing index cleared.")
    else:
        print("[VECTORSTORE] No index to clear.")

# ─────────────────────────────────────────────
# Smoke Test (Run directly: python retrieval.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion import ingest_documents
    import sys

    if len(sys.argv) < 2:
        print("Usage: python retrieval.py <file1> <file2> ...")
        sys.exit(1)

    file_paths = sys.argv[1:]

    # ── Step 1: Check if index already exists ──
    existing_vs = load_vectorstore()

    if existing_vs:
        print("✅ Using existing vector store.")
        vs = existing_vs
    else:
        print("🔄 Building new vector store...")
        docs = ingest_documents(file_paths)
        if not docs:
            print("[ERROR] No documents ingested. Exiting.")
            sys.exit(1)
        vs = build_vectorstore(docs)
        save_vectorstore(vs)

    # ── Step 2: Test Retrieval ──
    print("\n" + "="*50)
    print("RETRIEVAL + CONFIDENCE SCORING TEST")
    print("="*50)

    test_queries = [
        "What is Machine Learning?",           # Should be High confidence
        "What is supervised learning?",         # Should be High/Medium
        "What is the weather like today?",      # Should be Low → fallback
    ]

    for query in test_queries:
        print(f"\n{'─'*50}")
        result = retrieve_relevant_chunks(query, vs)
        print(f"Confidence  : {result['confidence']}")
        print(f"Should Answer: {result['should_answer']}")
        if result['chunks']:
            print(f"Top Chunk Preview: {result['chunks'][0].page_content[:200]}")