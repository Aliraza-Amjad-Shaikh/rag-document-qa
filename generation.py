from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List

from config import GOOGLE_API_KEY, CHAT_MODEL


# ─────────────────────────────────────────────
# Initialize Gemini LLM
# ─────────────────────────────────────────────

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize and return Gemini chat model via LangChain.
    """
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        max_output_tokens=1500
    )


# ─────────────────────────────────────────────
# Format Context
# ─────────────────────────────────────────────

def format_context(chunks: List[Document]) -> str:
    """
    Format retrieved chunks into a structured context block.
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
# Build Prompt
# ─────────────────────────────────────────────

def build_prompt(question: str, context: str) -> list:
    """
    Build LangChain messages list for Gemini.
    """
    system_content = """You are a precise and trustworthy document assistant.

Your ONLY job is to answer questions using the document context provided.

STRICT RULES:
1. Answer ONLY using information from the provided context.
2. ALWAYS cite your source using this format:
   - For PDFs:   (Source: filename.pdf | Page X)
   - For Images: (Source: imagename.png | Image)
3. If the answer is clearly present in the context, you MUST answer it fully.
4. Only say exactly "I don't know based on the provided documents." if the
   topic is completely absent from ALL provided context chunks.
5. NEVER fabricate or infer beyond what is written.
6. NEVER reference outside knowledge or training data.
7. Be thorough — use all relevant chunks to form your answer.
8. Format your answer clearly with bullet points or paragraphs as needed."""

    user_content = f"""Here is the relevant context retrieved from uploaded documents:

──────────────────────────────────────────────────────
{context}
──────────────────────────────────────────────────────

Based STRICTLY on the context above, answer this question:
QUESTION: {question}

Remember:
- If the answer exists in ANY context chunk above, you MUST provide it.
- Cite the exact source for every claim.
- Only say "I don't know based on the provided documents." if the topic
  is completely absent from the context.

ANSWER:"""

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ]


# ─────────────────────────────────────────────
# Fallback Detection
# ─────────────────────────────────────────────

def is_fallback_response(answer: str) -> bool:
    """
    Detect if the LLM genuinely could not answer.
    Uses exact phrase matching only — avoids false positives.
    """
    answer_lower = answer.lower().strip()
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
# Extract Sources
# ─────────────────────────────────────────────

def extract_sources(chunks: List[Document]) -> list:
    """
    Extract unique source citations from chunks.
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
    question:      str,
    chunks:        List[Document],
    confidence:    str,
    should_answer: bool
) -> dict:
    """
    Generate a grounded answer using Gemini.
    Triggers fallback if confidence is Low or no chunks found.
    """
    # Hard fallback
    if not should_answer or not chunks:
        print(f"[GENERATION] Fallback — should_answer={should_answer}, chunks={len(chunks)}")
        return {
            "answer":     "I don't know based on the provided documents.",
            "confidence": confidence,
            "sources":    [],
            "fallback":   True
        }

    context  = format_context(chunks)
    messages = build_prompt(question, context)

    print(f"[GENERATION] Sending {len(chunks)} chunks to Gemini...")
    print(f"[GENERATION] Context: {len(context)} characters")

    try:
        llm      = get_llm()
        response = llm.invoke(messages)
        answer   = response.content.strip()
        print(f"[GENERATION] ✅ Answer received: {answer[:100]}...")

    except Exception as e:
        print(f"[ERROR] Gemini generation failed: {e}")
        return {
            "answer":     "I don't know based on the provided documents.",
            "confidence": "Low",
            "sources":    [],
            "fallback":   True
        }

    sources  = extract_sources(chunks)
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
    vs         = get_or_build_vectorstore()

    if not vs:
        print("🔄 Building vector store...")
        docs = ingest_documents(file_paths)
        from retrieval import build_vectorstore, save_vectorstore
        vs   = build_vectorstore(docs)
        save_vectorstore(vs)

    test_queries = [
        "What is Machine Learning?",
        "What is the weather like today?",
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"❓ QUESTION: {query}")
        retrieval_result = retrieve_relevant_chunks(query, vs)
        result           = generate_answer(
            question=query,
            chunks=retrieval_result["chunks"],
            confidence=retrieval_result["confidence"],
            should_answer=retrieval_result["should_answer"]
        )
        print(f"💬 ANSWER    : {result['answer']}")
        print(f"📊 CONFIDENCE: {result['confidence']}")
        print(f"🔁 FALLBACK  : {result['fallback']}")