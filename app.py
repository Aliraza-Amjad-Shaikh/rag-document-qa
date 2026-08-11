import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import UPLOAD_DIR
from auth import validate_openai_key
from ingestion import save_uploaded_files, ingest_documents
from retrieval import (
    get_or_build_vectorstore,
    add_documents_to_vectorstore,
    remove_source_from_vectorstore,
    save_vectorstore,
    retrieve_relevant_chunks,
    clear_vectorstore,
    classify_query_type
)
from generation import generate_answer, generate_global_answer

from document_store import remove_document_entry, clear_document_store, load_document_store

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme: "reading room" — a library-card aesthetic for a document Q&A tool.
# Serif display for headings (Fraunces), clean sans for UI (Inter),
# monospace for data / filenames / citations (JetBrains Mono).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #0a0e14;
  --bg-alt: #0d131b;
  --surface: #121a24;
  --surface-2: #182230;
  --border: rgba(148, 163, 184, 0.14);
  --border-strong: rgba(148, 163, 184, 0.28);
  --text: #e6edf5;
  --text-muted: #8fa1b3;
  --accent: #f5b942;
  --accent-soft: rgba(245, 185, 66, 0.14);
  --accent-2: #56b6f0;
  --accent-2-soft: rgba(86, 182, 240, 0.14);
  --success: #34d399;
  --success-soft: rgba(52, 211, 153, 0.12);
  --warning: #f5b942;
  --warning-soft: rgba(245, 185, 66, 0.12);
  --danger: #f2637a;
  --danger-soft: rgba(242, 99, 122, 0.12);
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
}

/* ===== App shell ===== */
.stApp {
  background:
    radial-gradient(1200px 600px at 15% -10%, rgba(245, 185, 66, 0.05), transparent 55%),
    radial-gradient(1000px 500px at 100% 0%, rgba(86, 182, 240, 0.05), transparent 55%),
    var(--bg);
  color: var(--text);
}

.main .block-container {
  padding-top: 1.6rem;
  padding-bottom: 3rem;
  max-width: 980px;
}

h1, h2, h3 {
  font-family: 'Fraunces', serif !important;
  color: var(--text) !important;
  letter-spacing: -0.01em;
  font-weight: 600 !important;
}

hr {
  border-color: var(--border) !important;
  margin: 1.1rem 0 !important;
}

/* ===== Custom header ===== */
.rd-header {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  padding: 0.4rem 0 0.2rem 0;
}
.rd-header .rd-badge {
  width: 46px; height: 46px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem;
  background: linear-gradient(145deg, var(--accent-soft), var(--accent-2-soft));
  border: 1px solid var(--border-strong);
  flex-shrink: 0;
}
.rd-header .rd-title {
  font-family: 'Fraunces', serif;
  font-size: 1.7rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1.15;
}
.rd-header .rd-subtitle {
  color: var(--text-muted);
  font-size: 0.92rem;
  margin-top: 0.15rem;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--bg-alt), var(--bg)) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
section[data-testid="stSidebar"] .rd-sidebar-title {
  font-family: 'Fraunces', serif;
  font-size: 1.25rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
section[data-testid="stSidebar"] .rd-sidebar-caption {
  color: var(--text-muted) !important;
  font-size: 0.8rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-top: -0.2rem;
}
section[data-testid="stSidebar"] h3 {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.78rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted) !important;
  font-weight: 700 !important;
  margin-top: 0.4rem;
}

/* ===== File uploader ===== */
div[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border-strong) !important;
  border-radius: 14px !important;
  padding: 0.7rem !important;
  transition: border-color 0.15s ease, background 0.15s ease;
}
div[data-testid="stFileUploader"]:hover {
  border-color: var(--accent-2) !important;
  background: var(--surface-2) !important;
}
div[data-testid="stFileUploadDropzone"] p,
div[data-testid="stFileUploadDropzone"] span {
  color: var(--text-muted) !important;
}

/* ===== Buttons ===== */
.stButton > button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  border: 1px solid var(--border-strong) !important;
  background: linear-gradient(135deg, #d99a2b, #f5b942) !important;
  color: #1a1305 !important;
  padding: 0.55rem 1rem !important;
  transition: transform 0.12s ease, filter 0.12s ease, box-shadow 0.12s ease;
  box-shadow: 0 1px 0 rgba(255,255,255,0.15) inset;
}
.stButton > button:hover {
  transform: translateY(-1px);
  filter: brightness(1.06);
  box-shadow: 0 6px 16px rgba(245, 185, 66, 0.18);
}
.stButton > button:active { transform: translateY(0); }

button[kind="secondary"] {
  background: var(--danger-soft) !important;
  border: 1px solid rgba(242, 99, 122, 0.35) !important;
  color: #fca5b3 !important;
}

/* ===== Stat cards (replace default st.metric look) ===== */
.rd-stat-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.5rem;
  margin: 0.4rem 0 0.2rem 0;
}
.rd-stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.6rem 0.5rem;
  text-align: center;
}
.rd-stat-card .rd-stat-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--accent);
}
.rd-stat-card .rd-stat-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-top: 0.1rem;
}

/* ===== Status pill ===== */
.rd-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.4rem 0.75rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  border: 1px solid var(--border-strong);
  background: var(--surface);
}
.rd-status-pill .rd-dot {
  width: 8px; height: 8px; border-radius: 50%;
}
.rd-status-active .rd-dot { background: var(--success); box-shadow: 0 0 8px var(--success); }
.rd-status-idle .rd-dot { background: var(--text-muted); }

/* ===== Indexed file rows (library card look) ===== */
.rd-file-row {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.45rem 0.6rem;
  border-radius: 9px;
  border: 1px solid var(--border);
  background: var(--surface);
  margin-bottom: 0.35rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  border-left: 3px solid var(--accent-2);
}

/* ===== Metrics (fallback native, in case used elsewhere) ===== */
div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.7rem 0.8rem;
}

/* ===== Chat input ===== */
div[data-testid="stChatInput"] {
  background: rgba(10, 14, 20, 0.92) !important;
  border-top: 1px solid var(--border) !important;
  backdrop-filter: blur(6px);
}
div[data-testid="stChatInput"] textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
  font-family: 'Inter', sans-serif !important;
}
div[data-testid="stChatInput"] textarea:focus {
  border-color: var(--accent-2) !important;
  box-shadow: 0 0 0 3px var(--accent-2-soft) !important;
}

/* ===== Chat bubbles ===== */
div[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 0.3rem 0.2rem !important;
  margin-bottom: 0.6rem !important;
}

/* ===== Expanders (sources) ===== */
div[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
  overflow: hidden;
}
div[data-testid="stExpander"] summary {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.86rem !important;
}

/* ===== Citation chips (signature element) ===== */
.rd-citation {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.7rem 0.5rem 0.85rem;
  margin-bottom: 0.4rem;
  border-radius: 8px;
  background: var(--surface-2);
  border-top: 1px dashed var(--border-strong);
  border-left: 3px solid var(--accent);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--text);
}
.rd-citation .rd-citation-icon { font-size: 0.95rem; flex-shrink: 0; }
.rd-citation .rd-citation-meta { color: var(--text-muted); font-size: 0.76rem; }

/* ===== Confidence + fallback badges ===== */
.rd-badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.35rem 0; }
.rd-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.32rem 0.7rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  border: 1px solid var(--border-strong);
}
.rd-badge-high   { background: var(--success-soft); color: #6ee7b7; }
.rd-badge-medium { background: var(--warning-soft); color: #fcd34d; }
.rd-badge-low    { background: var(--danger-soft);  color: #fca5b3; }
.rd-badge-fallback { background: var(--danger-soft); color: #fca5b3; }

/* ===== Empty state / intro cards ===== */
.rd-intro-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem 1rem;
  height: 100%;
}
.rd-intro-card .rd-intro-icon { font-size: 1.4rem; margin-bottom: 0.35rem; }
.rd-intro-card .rd-intro-title {
  font-family: 'Fraunces', serif;
  font-weight: 600;
  font-size: 1rem;
  margin-bottom: 0.25rem;
}
.rd-intro-card .rd-intro-body {
  color: var(--text-muted);
  font-size: 0.85rem;
  line-height: 1.4;
}

.rd-empty-chat {
  text-align: center;
  color: var(--text-muted);
  padding: 2.2rem 1rem;
  border: 1px dashed var(--border-strong);
  border-radius: 14px;
  background: var(--surface);
  margin: 0.5rem 0 1rem 0;
}

/* ===== Success/Warning/Error native boxes (auth screen etc.) ===== */
div[data-testid="stAlert"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}

/* ===== Auth gate card ===== */
.rd-auth-card {
  max-width: 520px;
  margin: 2.2rem auto 0 auto;
  background: var(--surface);
  border: 1px solid var(--border-strong);
  border-radius: 18px;
  padding: 2rem 2.1rem;
  text-align: center;
}
.rd-auth-card .rd-auth-icon {
  font-size: 2rem;
  margin-bottom: 0.6rem;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Small display helpers (presentation only — no logic changes) ──

def render_confidence_badge(confidence: str, fallback: bool = False):
    conf_class = {
        "High": "rd-badge-high",
        "Medium": "rd-badge-medium",
    }.get(confidence, "rd-badge-low")
    conf_icon = {"High": "🟢", "Medium": "🟡"}.get(confidence, "🔴")

    html = f'<div class="rd-badge-row"><span class="rd-badge {conf_class}">{conf_icon} Confidence: {confidence}</span>'
    if fallback:
        html += '<span class="rd-badge rd-badge-fallback">⚠️ Fallback — not found in documents</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_sources(sources):
    if not sources:
        return
    with st.expander("📚 Sources"):
        for src in sources:
            if src["type"] == "pdf":
                st.markdown(
                    f'<div class="rd-citation"><span class="rd-citation-icon">📄</span>'
                    f'<span>{src["source"]}</span>'
                    f'<span class="rd-citation-meta">· page {src["page_number"]}</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="rd-citation"><span class="rd-citation-icon">🖼️</span>'
                    f'<span>{src["image_name"]}</span>'
                    f'<span class="rd-citation-meta">· image</span></div>',
                    unsafe_allow_html=True
                )


# ── Session State ──
defaults = {
    "openai_api_key": None,
    "api_key_validated": False,
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

# ── API KEY GATE ──

if not st.session_state.api_key_validated:
    st.markdown(
        """
        <div class="rd-auth-card">
            <div class="rd-auth-icon">🔐</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("OpenAI API Key Required")
    st.write(
        "Enter your OpenAI API key to use the document Q&A system."
    )
    st.caption(
        "Your key is used only for this Streamlit session and is not saved by the app."
    )

    entered_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="The key is required to process documents and answer questions.",
    )

    if st.button("Validate API Key", type="primary"):
        if not entered_api_key.strip():
            st.error("Please enter an API key.")
        else:
            with st.spinner("Validating API key..."):
                is_valid, message = validate_openai_key(
                    entered_api_key.strip()
                )

            if is_valid:
                st.session_state.openai_api_key = entered_api_key.strip()
                st.session_state.api_key_validated = True
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    st.stop()

api_key = st.session_state.openai_api_key

# ── SIDEBAR ──
with st.sidebar:
    st.markdown(
        """
        <div class="rd-sidebar-title">📚 RAG Document Q&A</div>
        <div class="rd-sidebar-caption">Retrieval-Augmented Generation</div>
        """,
        unsafe_allow_html=True
    )
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

            existing_store = load_document_store()
            existing_vs = get_or_build_vectorstore(api_key=api_key)
            if existing_vs:
                for f in uploaded_files:
                    if f.name in existing_store:
                        existing_vs = remove_source_from_vectorstore(existing_vs, f.name)
                        remove_document_entry(f.name)
                    save_vectorstore(existing_vs)

            with st.spinner("Extracting content..."):
                docs = ingest_documents(saved_paths, api_key=api_key)

            if docs:
                pdf_chunks   = len([d for d in docs if d.metadata.get("type") == "pdf"])
                image_chunks = len([d for d in docs if d.metadata.get("type") == "image"])

                with st.spinner("Building vector store..."):
                    vs = add_documents_to_vectorstore(documents=docs, api_key=api_key)

                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.is_ready = True

                    new_files = [f.name for f in uploaded_files if f.name not in st.session_state.indexed_files]
                    replaced_files = [f.name for f in uploaded_files if f.name in st.session_state.indexed_files]

                    st.session_state.indexed_files.extend(new_files)

                    new_docs = [d for d in docs if d.metadata.get("source") in new_files]
                    st.session_state.total_chunks += len(new_docs)
                    st.session_state.pdf_chunks += len([d for d in new_docs if d.metadata.get("type") == "pdf"])
                    st.session_state.image_chunks += len([d for d in new_docs if d.metadata.get("type") == "image"])
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
        st.markdown(
            '<div class="rd-status-pill rd-status-active"><span class="rd-dot"></span>Vector store active</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">'
            f'{len(st.session_state.indexed_files)} file(s) loaded</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="rd-stat-grid">
                <div class="rd-stat-card">
                    <div class="rd-stat-value">{st.session_state.total_chunks}</div>
                    <div class="rd-stat-label">Total</div>
                </div>
                <div class="rd-stat-card">
                    <div class="rd-stat-value">{st.session_state.pdf_chunks}</div>
                    <div class="rd-stat-label">PDF</div>
                </div>
                <div class="rd-stat-card">
                    <div class="rd-stat-value">{st.session_state.image_chunks}</div>
                    <div class="rd-stat-label">Image</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()
        st.subheader("📂 Indexed Files")
        for fname in st.session_state.indexed_files:
            ext  = os.path.splitext(fname)[1].lower()
            icon = "📄" if ext == ".pdf" else "🖼️"
            st.markdown(
                f'<div class="rd-file-row">{icon} {fname}</div>',
                unsafe_allow_html=True
            )

        if st.button("🔒 Remove API Key"):
            st.session_state.openai_api_key = None
            st.session_state.api_key_validated = False
            st.session_state.vectorstore = None
            st.session_state.is_ready = False
            st.rerun()

        st.divider()
        if st.button("🗑️ Clear & Start Over"):
            clear_vectorstore()
            clear_document_store()
            for key, value in defaults.items():
                if key not in {"openai_api_key", "api_key_validated"}:
                    st.session_state[key] = value
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
        st.markdown(
            '<div class="rd-status-pill rd-status-idle"><span class="rd-dot"></span>No documents loaded</div>',
            unsafe_allow_html=True
        )
        st.write("Upload files to begin")

# ── MAIN AREA ──
st.markdown(
    """
    <div class="rd-header">
        <div class="rd-badge">📚</div>
        <div>
            <div class="rd-title">RAG Document Q&A</div>
            <div class="rd-subtitle">Upload PDFs or images · Ask questions · Get cited answers</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

if not st.session_state.is_ready:
    st.info("👈 Upload and process documents using the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="rd-intro-card">
                <div class="rd-intro-icon">📄</div>
                <div class="rd-intro-title">PDF Support</div>
                <div class="rd-intro-body">Multi-mode text extraction with page citations</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="rd-intro-card">
                <div class="rd-intro-icon">🖼️</div>
                <div class="rd-intro-title">Image AI</div>
                <div class="rd-intro-body">GPT-4o Vision reads charts, tables and diagrams</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="rd-intro-card">
                <div class="rd-intro-icon">🎯</div>
                <div class="rd-intro-title">No Hallucination</div>
                <div class="rd-intro-body">Confidence scoring prevents fabricated answers</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.stop()

# Chat area
if not st.session_state.messages:
    st.markdown(
        '<div class="rd-empty-chat">💬 Documents ready — ask a question below</div>',
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant":
            conf     = msg.get("confidence", "Low")
            fallback = msg.get("fallback", False)
            sources  = msg.get("sources", [])

            render_confidence_badge(conf, fallback)
            render_sources(sources)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                query_type = classify_query_type(prompt)

                if query_type == "global":
                    result = retrieve_relevant_chunks(prompt, st.session_state.vectorstore, api_key=api_key)
                else:
                    retrieval_result = retrieve_relevant_chunks(prompt, st.session_state.vectorstore, api_key=api_key)
                    result = generate_answer(
                        question=prompt,
                        chunks=retrieval_result["chunks"],
                        confidence=retrieval_result["confidence"],
                        should_answer=retrieval_result["should_answer"],
                        api_key=api_key
                    )

                st.write(result["answer"])

                conf = result["confidence"]
                fallback = result["fallback"]
                sources = result["sources"]

                render_confidence_badge(conf, fallback)
                render_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "confidence": conf,
                    "sources": sources,
                    "fallback": fallback
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}",
                    "confidence": "Low",
                    "sources": [],
                    "fallback": True
                })