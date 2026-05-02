from openai import OpenAI
from langchain_core.documents import Document
from config import OPENAI_API_KEY, CHAT_MODEL

# ─────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────
# Format Retrieved Chunks into Context Block
# ─────────────────────────────────────────────

def format_context(chunks: list[Document]) -> str:
    """
    Format retrieved document chunks into a structured
    context block for the LLM prompt.

    Args:
        chunks: List of retrieved Document objects

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        source   = chunk.metadata.get("source", "Unknown")
        doc_type = chunk.metadata.get("type", "unknown")

        if doc_type == "pdf":
            page         = chunk.metadata.get("page_number", "?")
            source_label = f"[Source {i+1}: {source} | Page {page}]"
        elif doc_type == "image":
            image_name   = chunk.metadata.get("image_name", source)
            source_label = f"[Source {i+1}: {image_name} | Image]"
        else:
            source_label = f"[Source {i+1}: {source}]"

        context_parts.append(
            f"{source_label}\n{chunk.page_content.strip()}"
        )

    return "\n\n".join(context_parts)


# ─────────────────────────────────────────────
# Build the LLM Prompt
# ─────────────────────────────────────────────

def build_prompt(question: str, context: str) -> list[dict]:
    """
    Build the messages list for OpenAI chat completion.
    Enforces strict grounding while being helpful.

    Args:
        question: User's natural language question
        context:  Formatted context from format_context()

    Returns:
        List of message dicts for OpenAI API
    """
    system_prompt = """You are a precise and trustworthy document assistant.

Your ONLY job is to answer questions using the document context provided below.

STRICT RULES:
1. Answer ONLY using information from the provided context.
2. ALWAYS cite your source using this format:
   - For PDFs: (Source: filename.pdf | Page X)
   - For Images: (Source: imagename.png | Image)
3. If the answer is clearly present in the context, you MUST answer it fully.
4. Only say "I don't know based on the provided documents." if the topic is 
   completely absent from ALL provided context chunks.
5. NEVER fabricate or infer beyond what is written.
6. NEVER reference outside knowledge or training data.
7. Be thorough — if multiple chunks support the answer, use all of them.
8. Format your answer clearly with bullet points or paragraphs as appropriate."""

    user_message = f"""Here is the relevant context retrieved from the uploaded documents:

──────────────────────────────────────────────────────
{context}
──────────────────────────────────────────────────────

Based STRICTLY on the context above, answer this question:
QUESTION: {question}

Remember:
- If the answer exists in ANY of the context chunks above, you MUST provide it.
- Cite the exact source for every claim you make.
- Only respond with "I don't know based on the provided documents." if the 
  topic is completely absent from the context.

ANSWER:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message}
    ]


# ─────────────────────────────────────────────
# Check if LLM Genuinely Could Not Answer
# ─────────────────────────────────────────────

def is_fallback_response(answer: str) -> bool:
    """
    Detect if the LLM genuinely could not answer
    using EXACT phrase matching only.
    Avoids false positives from partial phrase matches.

    Args:
        answer: LLM generated answer string

    Returns:
        True if this is a genuine fallback response
    """
    answer_lower = answer.lower().strip()

    # Only trigger fallback on these EXACT phrases
    fallback_phrases = [
        "i don't know based on the provided documents",
        "i do not know based on the provided documents",
        "the provided documents do not contain",
        "the context does not contain",
        "not mentioned in the provided documents",
        "no information available in the provided documents",
    ]

    return any(phrase in answer_lower for phrase in fallback_phrases)


# ─────────────────────────────────────────────
# Extract Source Citations
# ─────────────────────────────────────────────

def extract_sources(chunks: list[Document]) -> list[dict]:
    """
    Extract unique source citations from retrieved chunks.

    Args:
        chunks: List of retrieved Document objects

    Returns:
        List of unique source citation dicts
    """
    seen    = set()
    sources = []

    for chunk in chunks:
        doc_type = chunk.metadata.get("type", "unknown")
        source   = chunk.metadata.get("source", "Unknown")

        if doc_type == "pdf":
            page = chunk.metadata.get("page_number", "?")
            key  = f"{source}_page_{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source":      source,
                    "type":        "pdf",
                    "page_number": page
                })

        elif doc_type == "image":
            key = f"{source}_image"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source":     source,
                    "type":       "image",
                    "image_name": chunk.metadata.get("image_name", source)
                })

    return sources


# ─────────────────────────────────────────────
# Generate Answer
# ─────────────────────────────────────────────

def generate_answer(
    question:     str,
    chunks:       list[Document],
    confidence:   str,
    should_answer: bool
) -> dict:
    """
    Generate a grounded answer using the LLM.
    Triggers fallback ONLY when confidence is Low
    or no chunks were retrieved.

    Args:
        question:      User's natural language question
        chunks:        Retrieved Document chunks
        confidence:    Confidence label (High/Medium/Low)
        should_answer: Boolean from retrieval pipeline

    Returns:
        dict with keys:
            - answer, confidence, sources, fallback
    """
    # ── Hard fallback: No chunks or Low confidence ──
    if not should_answer or not chunks:
        print(f"[GENERATION] Fallback triggered — "
              f"should_answer={should_answer}, chunks={len(chunks)}")
        return {
            "answer":     "I don't know based on the provided documents.",
            "confidence": confidence,
            "sources":    [],
            "fallback":   True
        }

    # ── Format context and build prompt ──
    context  = format_context(chunks)
    messages = build_prompt(question, context)

    print(f"[GENERATION] Sending {len(chunks)} chunks to LLM...")
    print(f"[GENERATION] Context length: {len(context)} characters")

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1500
        )
        answer = response.choices[0].message.content.strip()
        print(f"[GENERATION] Answer received: {answer[:100]}...")

    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        return {
            "answer":     "I don't know based on the provided documents.",
            "confidence": "Low",
            "sources":    [],
            "fallback":   True
        }

    # ── Extract sources ──
    sources = extract_sources(chunks)

    # ── Check if LLM itself said it doesn't know ──
    fallback = is_fallback_response(answer)

    return {
        "answer":     answer,
        "confidence": confidence,
        "sources":    sources,
        "fallback":   fallback
    }


# ─────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from retrieval import get_or_build_vectorstore, retrieve_relevant_chunks
    from ingestion import ingest_documents
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generation.py <file1> <file2> ...")
        sys.exit(1)

    file_paths = sys.argv[1:]

    vs = get_or_build_vectorstore()
    if not vs:
        print("🔄 Building vector store...")
        docs = ingest_documents(file_paths)
        from retrieval import build_vectorstore, save_vectorstore
        vs = build_vectorstore(docs)
        save_vectorstore(vs)

    test_queries = [
        "What is scikit learn?",
        "What is supervised learning?",
        "What is the weather like today?",
    ]

    print("\n" + "="*60)
    print("LLM ANSWER GENERATION TEST")
    print("="*60)

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"❓ QUESTION: {query}")

        retrieval_result = retrieve_relevant_chunks(query, vs)
        result = generate_answer(
            question=query,
            chunks=retrieval_result["chunks"],
            confidence=retrieval_result["confidence"],
            should_answer=retrieval_result["should_answer"]
        )

        print(f"\n💬 ANSWER:\n{result['answer']}")
        print(f"\n📊 CONFIDENCE : {result['confidence']}")
        print(f"🔁 FALLBACK   : {result['fallback']}")
        print(f"\n📚 SOURCES:")
        for s in result["sources"]:
            if s["type"] == "pdf":
                print(f"   - {s['source']} | Page {s['page_number']}")
            else:
                print(f"   - {s['image_name']} | Image")