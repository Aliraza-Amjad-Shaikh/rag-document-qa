import streamlit as st
import os
import sys

# Ensure local modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import UPLOAD_DIR, VECTORSTORE_DIR, SUPPORTED_EXTENSIONS
from ingestion import save_uploaded_files, ingest_documents
from retrieval import get_or_build_vectorstore, retrieve_relevant_chunks, clear_vectorstore
from generation import generate_answer

# ─────────────────────────────────────────────
# Page Configuration & Custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern UI Styling
st.markdown("""
<style>
    /* Base App Styling */
    .stApp { background-color: #f8f9fa; }
    .main-title { font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; }
    .subtitle { color: #64748b; margin-bottom: 1.5rem; }

    /* Chat Bubbles */
    .user-msg {
        background: linear-gradient(135deg, #e0f2fe, #bae6fd);
        padding: 12px 16px;
        border-radius: 12px 12px 0 12px;
        margin: 8px 0 16px 0;
        border-left: 4px solid #0284c7;
        color: #0f172a;
    }
    .assistant-msg {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px 12px 12px 0;
        margin: 8px 0 16px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        color: #1e293b;
    }

    /* Confidence Badges */
    .conf-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    .conf-high { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .conf-medium { background-color: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .conf-low { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }

    /* Source Citations Box */
    .source-box {
        background-color: #f1f5f9;
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 12px;
        border-left: 4px solid #64748b;
        font-size: 0.9rem;
    }
    .source-box ul { margin: 8px 0 0 18px; padding: 0; }
    .source-box li { margin-bottom: 4px; color: #334155; }
    .source-box code { background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-family: monospace; }

    /* Sidebar Elements */
    .sidebar-card {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .status-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 0.9rem; }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
    .dot-green { background-color: #22c55e; box-shadow: 0 0 6px #22c55e; }
    .dot-red { background-color: #ef4444; box-shadow: 0 0 6px #ef4444; }
    .dot-yellow { background-color: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Management
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ─────────────────────────────────────────────
# Sidebar: Upload & Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Q&A System")
    st.markdown("---")

    # Upload Card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDFs or Images",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Supports text-based PDFs and standard image formats"
    )

    if uploaded_files:
        new_files = [f.name for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            if st.button("🚀 Process & Index", type="primary", use_container_width=True):
                with st.spinner("💾 Saving files to disk..."):
                    saved_paths = save_uploaded_files(uploaded_files)

                with st.spinner("📖 Extracting & Chunking documents..."):
                    docs = ingest_documents(saved_paths)

                if docs:
                    # Show what was found
                    pdf_chunks   = len([d for d in docs if d.metadata.get("type") == "pdf"])
                    image_chunks = len([d for d in docs if d.metadata.get("type") == "image"])
                    st.info(f"📄 PDF chunks: {pdf_chunks} | 🖼️ Image chunks: {image_chunks}")

                    with st.spinner("🧠 Embedding & Building Vector Store..."):
                        # Always pass docs so it rebuilds fresh
                        vs = get_or_build_vectorstore(documents=docs)

                        if vs:
                            st.session_state.vectorstore   = vs
                            st.session_state.is_ready      = True
                            st.session_state.indexed_files = [f.name for f in uploaded_files]
                            st.success(
                                f"✅ Successfully indexed {len(docs)} chunks "
                                f"from {len(uploaded_files)} file(s)!"
                            )
                            st.rerun()
                        else:
                            st.error("❌ Failed to build vector store.")
                else:
                    st.warning("⚠️ No valid content extracted from documents.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Status Card
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.subheader("📊 System Status")
    if st.session_state.is_ready:
        st.markdown(f'<div class="status-row"><div class="dot dot-green"></div> Vector Store Loaded</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-row"><div class="dot dot-green"></div> {len(st.session_state.indexed_files)} File(s) Indexed</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-row"><div class="dot dot-red"></div> No Documents Loaded</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Reset Button
    if st.session_state.is_ready:
        if st.button("🗑️ Clear & Upload New", type="secondary", use_container_width=True):
            clear_vectorstore()
            st.session_state.vectorstore = None
            st.session_state.is_ready = False
            st.session_state.indexed_files = []
            st.session_state.messages = []
            # Clean uploads folder
            if os.path.exists(UPLOAD_DIR):
                for f in os.listdir(UPLOAD_DIR):
                    os.remove(os.path.join(UPLOAD_DIR, f))
            st.success("🧹 System cleared! Ready for new documents.")
            st.rerun()

# ─────────────────────────────────────────────
# Main Chat Interface
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-title">💬 Ask Your Documents</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload PDFs or images, then ask questions. Answers include source citations and confidence scoring.</p>', unsafe_allow_html=True)

if not st.session_state.is_ready:
    st.info("👈 Upload and process documents using the sidebar to get started.")
    st.stop()

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            
            # Confidence Badge
            conf = msg.get("confidence", "Low")
            conf_class = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(conf, "conf-low")
            conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "🔴")
            st.markdown(f'<div class="conf-badge {conf_class}">{conf_emoji} Confidence: {conf}</div>', unsafe_allow_html=True)

            # Source Citations
            if msg.get("sources"):
                sources_html = '<div class="source-box"><strong>📚 Sources Cited:</strong><ul>'
                for src in msg["sources"]:
                    if src["type"] == "pdf":
                        sources_html += f'<li><code>{src["source"]}</code> | Page {src["page_number"]}</li>'
                    elif src["type"] == "image":
                        sources_html += f'<li><code>{src["image_name"]}</code> | Image</li>'
                sources_html += '</ul></div>'
                st.markdown(sources_html, unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving context & generating answer..."):
            try:
                retrieval_result = retrieve_relevant_chunks(prompt, st.session_state.vectorstore)
                result = generate_answer(
                    question=prompt,
                    chunks=retrieval_result["chunks"],
                    confidence=retrieval_result["confidence"],
                    should_answer=retrieval_result["should_answer"]
                )

                # Display response
                st.markdown(f'<div class="assistant-msg">{result["answer"]}</div>', unsafe_allow_html=True)
                
                conf = result["confidence"]
                conf_class = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(conf, "conf-low")
                conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "🔴")
                st.markdown(f'<div class="conf-badge {conf_class}">{conf_emoji} Confidence: {conf}</div>', unsafe_allow_html=True)

                if result["sources"]:
                    sources_html = '<div class="source-box"><strong>📚 Sources Cited:</strong><ul>'
                    for src in result["sources"]:
                        if src["type"] == "pdf":
                            sources_html += f'<li><code>{src["source"]}</code> | Page {src["page_number"]}</li>'
                        elif src["type"] == "image":
                            sources_html += f'<li><code>{src["image_name"]}</code> | Image</li>'
                    sources_html += '</ul></div>'
                    st.markdown(sources_html, unsafe_allow_html=True)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "confidence": conf,
                    "sources": result["sources"]
                })

            except Exception as e:
                st.error(f"⚠️ An error occurred: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I encountered an error while processing your question. Please try again.",
                    "confidence": "Low",
                    "sources": []
                })