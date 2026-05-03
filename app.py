import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import UPLOAD_DIR
from ingestion import save_uploaded_files, ingest_documents
from retrieval import get_or_build_vectorstore, retrieve_relevant_chunks, clear_vectorstore
from generation import generate_answer

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ===== Safe Dark Theme for Streamlit ===== */

/* App background */
.stApp {
  background-color: #0b1220;
  color: #e6edf3;
}

/* Main container spacing */
.main .block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1000px;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
  background-color: #0f172a !important;
  border-right: 1px solid rgba(148, 163, 184, 0.15);
}

/* Sidebar content text */
section[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
}

/* Headings */
h1, h2, h3 {
  color: #e6edf3 !important;
  letter-spacing: -0.02em;
}

/* Streamlit divider */
hr {
  border-color: rgba(148, 163, 184, 0.18) !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px dashed rgba(148, 163, 184, 0.35) !important;
  border-radius: 12px !important;
  padding: 0.6rem !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: rgba(96,165,250,0.9) !important;
}
div[data-testid="stFileUploadDropzone"] p {
  color: rgba(226,232,240,0.75) !important;
}

/* Buttons */
.stButton > button {
  border-radius: 12px !important;
  font-weight: 700 !important;
  border: 1px solid rgba(148, 163, 184, 0.18) !important;
  background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
  color: white !important;
  padding: 0.6rem 1rem !important;
  transition: transform 0.12s ease, filter 0.12s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  filter: brightness(1.05);
}

/* Secondary buttons (like Clear) - Streamlit may render differently, keep generic */
button[kind="secondary"] {
  background: rgba(248,113,113,0.08) !important;
  border: 1px solid rgba(248,113,113,0.35) !important;
  color: #fecaca !important;
}

/* Metrics */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(148, 163, 184, 0.15);
  border-radius: 12px;
  padding: 0.7rem 0.8rem;
}

/* Chat input container */
div[data-testid="stChatInput"] {
  background: rgba(15,23,42,0.9) !important;
  border-top: 1px solid rgba(148, 163, 184, 0.15) !important;
}

/* Chat textarea */
div[data-testid="stChatInput"] textarea {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(148, 163, 184, 0.18) !important;
  color: #e5e7eb !important;
  border-radius: 12px !important;
}
div[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(96,165,250,0.95) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.18) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
  border: 1px solid rgba(148, 163, 184, 0.15) !important;
  border-radius: 12px !important;
  background: rgba(255,255,255,0.02) !important;
}

/* Success/Warning/Error boxes */
div[data-testid="stAlert"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(148, 163, 184, 0.15) !important;
  border-radius: 12px !important;
  color: #e5e7eb !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
defaults = {
    "messages":      [],
    "vectorstore":   None,
    "is_ready":      False,
    "indexed_files": [],
    "total_chunks":  0,
    "pdf_chunks":    0,
    "image_chunks":  0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ──
with st.sidebar:
    st.title("📚 RAG Document Q&A")
    st.caption("Retrieval-Augmented Generation System")
    st.divider()

    # Upload
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or Images",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected")

        if st.button("🚀 Process & Index", type="primary"):
            with st.spinner("Saving files..."):
                saved_paths = save_uploaded_files(uploaded_files)

            with st.spinner("Extracting content..."):
                docs = ingest_documents(saved_paths)

            if docs:
                pdf_chunks   = len([d for d in docs if d.metadata.get("type") == "pdf"])
                image_chunks = len([d for d in docs if d.metadata.get("type") == "image"])

                with st.spinner("Building vector store..."):
                    vs = get_or_build_vectorstore(documents=docs)

                if vs:
                    st.session_state.vectorstore   = vs
                    st.session_state.is_ready      = True
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.session_state.total_chunks  = len(docs)
                    st.session_state.pdf_chunks    = pdf_chunks
                    st.session_state.image_chunks  = image_chunks
                    st.success(f"✅ {len(docs)} chunks indexed!")
                    st.rerun()
                else:
                    st.error("❌ Failed to build vector store.")
            else:
                st.warning("⚠️ No content extracted.")

    st.divider()

    # Status
    st.subheader("⚡ Status")
    if st.session_state.is_ready:
        st.success("Vector store active")
        st.info(f"{len(st.session_state.indexed_files)} file(s) loaded")
        st.metric("Total Chunks", st.session_state.total_chunks)
        st.metric("PDF Chunks",   st.session_state.pdf_chunks)
        st.metric("Image Chunks", st.session_state.image_chunks)

        st.divider()
        st.subheader("📂 Indexed Files")
        for fname in st.session_state.indexed_files:
            ext  = os.path.splitext(fname)[1].lower()
            icon = "📄" if ext == ".pdf" else "🖼️"
            st.write(f"{icon} {fname}")

        st.divider()
        if st.button("🗑️ Clear & Start Over"):
            clear_vectorstore()
            for k, v in defaults.items():
                st.session_state[k] = v
            if os.path.exists(UPLOAD_DIR):
                for f in os.listdir(UPLOAD_DIR):
                    try:
                        os.remove(os.path.join(UPLOAD_DIR, f))
                    except Exception:
                        pass
            st.rerun()
    else:
        st.warning("No documents loaded")
        st.write("Upload files to begin")

# ── MAIN AREA ──
st.title("📚 RAG Document Q&A")
st.caption("Upload PDFs or images · Ask questions · Get cited answers")
st.divider()

if not st.session_state.is_ready:
    st.info("👈 Upload and process documents using the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📄 **PDF Support**\n\nMulti-mode text extraction with page citations")
    with col2:
        st.info("🖼️ **Image AI**\n\nGPT-4o Vision reads charts, tables and diagrams")
    with col3:
        st.info("🎯 **No Hallucination**\n\nConfidence scoring prevents fabricated answers")
    st.stop()

# Chat area
if not st.session_state.messages:
    st.write("")
    st.markdown(
        "<div style='text-align:center; color:gray; padding: 2rem'>💬 Documents ready — ask a question below</div>",
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant":
            conf     = msg.get("confidence", "Low")
            fallback = msg.get("fallback", False)
            sources  = msg.get("sources", [])

            if conf == "High":
                st.success(f"🟢 Confidence: {conf}")
            elif conf == "Medium":
                st.warning(f"🟡 Confidence: {conf}")
            else:
                st.error(f"🔴 Confidence: {conf}")

            if fallback:
                st.error("⚠️ Fallback triggered — answer not found in documents")

            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        if src["type"] == "pdf":
                            st.write(f"📄 `{src['source']}` — Page {src['page_number']}")
                        else:
                            st.write(f"🖼️ `{src['image_name']}` — Image")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                retrieval_result = retrieve_relevant_chunks(
                    prompt,
                    st.session_state.vectorstore
                )
                result = generate_answer(
                    question=prompt,
                    chunks=retrieval_result["chunks"],
                    confidence=retrieval_result["confidence"],
                    should_answer=retrieval_result["should_answer"]
                )

                st.write(result["answer"])

                conf     = result["confidence"]
                fallback = result["fallback"]
                sources  = result["sources"]

                if conf == "High":
                    st.success(f"🟢 Confidence: {conf}")
                elif conf == "Medium":
                    st.warning(f"🟡 Confidence: {conf}")
                else:
                    st.error(f"🔴 Confidence: {conf}")

                if fallback:
                    st.error("⚠️ Fallback triggered")

                if sources:
                    with st.expander("📚 Sources"):
                        for src in sources:
                            if src["type"] == "pdf":
                                st.write(f"📄 `{src['source']}` — Page {src['page_number']}")
                            else:
                                st.write(f"🖼️ `{src['image_name']}` — Image")

                st.session_state.messages.append({
                    "role":       "assistant",
                    "content":    result["answer"],
                    "confidence": conf,
                    "sources":    sources,
                    "fallback":   fallback
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({
                    "role":       "assistant",
                    "content":    f"Error: {str(e)}",
                    "confidence": "Low",
                    "sources":    [],
                    "fallback":   True
                })