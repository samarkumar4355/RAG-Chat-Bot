"""
📄 RAG Document Q&A — Clean ChatGPT-Style Interface.
Pre-configured with free models (Groq, OpenRouter, Gemini).
No API key entry required in the UI.
Fixed: Text overlapping, z-index layering, and smooth sidebar UX.
"""

import os
import tempfile
import streamlit as st

from rag_engine.pdf_processor import PDFProcessor
from rag_engine.vector_store import VectorStoreManager
from rag_engine.qa_chain import QAChain
from rag_engine.config import API_KEYS, AVAILABLE_FREE_MODELS, DEFAULT_MODEL

# ── Paths ──────────────────────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling (Clean, Robust, No Overlaps) ─────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, .stApp, .stMarkdown, p, h1, h2, h3, h4, input, textarea, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Preserve Material Symbols font for all Streamlit icons */
    [data-testid*="Icon"],
    [data-testid="stIconMaterial"],
    [class*="material-symbols"],
    span[class*="material"],
    button span:has(svg) {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* Root App Background */
    .stApp {
        background-color: #060814 !important;
    }

    /* Keep the native sidebar toggle while hiding unrelated toolbar actions. */
    #MainMenu, footer,
    .stDeployButton,
    [data-testid="stToolbarActions"],
    [data-testid="stAppDeployButton"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] { display: none !important; }

    /* Header & Sidebar Toggle Buttons UX */
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
        z-index: 1000 !important;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }

    /* ── BUTTON TO OPEN SIDEBAR (When closed) ── */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: 14px !important;
        left: 16px !important;
        z-index: 99999 !important;
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"] button {
        background: #0f1428 !important;
        border: 1px solid rgba(0, 214, 255, 0.45) !important;
        border-radius: 20px !important;
        color: #ffffff !important;
        padding: 6px 14px !important;
        box-shadow: 0 4px 18px rgba(0, 80, 255, 0.35) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stExpandSidebarButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {
        background: rgba(0, 80, 255, 0.3) !important;
        border-color: #00D6FF !important;
        box-shadow: 0 0 20px rgba(0, 214, 255, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stExpandSidebarButton"]::after,
    [data-testid="stExpandSidebarButton"] button::after,
    [data-testid="stSidebarCollapsedControl"] button::after {
        content: " ☰ Open Sidebar" !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        letter-spacing: -0.01em !important;
    }

    /* ── BUTTON TO CLOSE SIDEBAR (When open) ── */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stSidebarCollapseButton"] button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        padding: 4px 10px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 4px !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover {
        background: rgba(239, 68, 68, 0.25) !important;
        border-color: #ef4444 !important;
        color: #ef4444 !important;
    }
    [data-testid="stSidebarCollapseButton"] button::after {
        content: " Close ✕" !important;
        font-size: 11px !important;
        font-weight: 600 !important;
    }

    /* Native Block Container (Main Content Area) */
    .main .block-container {
        max-width: 840px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 180px !important; /* Prevents any overlap with bottom chat input */
        margin: 0 auto !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #0c0f1d !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: rgba(255, 255, 255, 0.8) !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        font-size: 17px;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }

    .brand-glow {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0050FF, #00D6FF);
        box-shadow: 0 0 14px #00D6FF;
    }

    .sidebar-section-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255, 255, 255, 0.45) !important;
        margin: 18px 0 8px 0;
    }

    .sidebar-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }

    /* Ambient animated background (Deep behind all content) */
    .animated-bg {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
    }
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(100px);
        opacity: 0.22;
        animation: orbFloat 24s ease-in-out infinite;
    }
    .orb-1 {
        width: 450px; height: 450px;
        background: radial-gradient(circle, #0050FF 0%, transparent 70%);
        top: -5%; left: 20%;
    }
    .orb-2 {
        width: 400px; height: 400px;
        background: radial-gradient(circle, #00D6FF 0%, transparent 70%);
        bottom: 5%; right: 10%;
        animation-delay: -10s;
    }
    .orb-3 {
        width: 320px; height: 320px;
        background: radial-gradient(circle, #6366f1 0%, transparent 70%);
        top: 50%; left: 35%;
        animation-delay: -18s;
        opacity: 0.15;
    }

    @keyframes orbFloat {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33%      { transform: translate(40px, -30px) scale(1.08); }
        66%      { transform: translate(-30px, 40px) scale(0.94); }
    }

    .grid-overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.012) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.012) 1px, transparent 1px);
        background-size: 50px 50px;
        z-index: 0;
        pointer-events: none;
    }

    /* Minimal Top Status Pill */
    .chat-top-status {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 16px;
        margin-bottom: 24px;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }

    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8);
    }

    .status-dot-green {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 8px #22c55e;
    }

    /* Empty State Hero */
    .chat-empty-hero {
        text-align: center;
        padding: 60px 20px 40px 20px;
    }

    .chat-empty-icon {
        width: 64px; height: 64px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(0, 80, 255, 0.2), rgba(0, 214, 255, 0.15));
        border: 1px solid rgba(0, 80, 255, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin: 0 auto 18px auto;
        box-shadow: 0 8px 30px rgba(0, 80, 255, 0.2);
    }

    .chat-empty-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .chat-empty-subtitle {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.5);
        max-width: 480px;
        margin: 0 auto 30px auto;
        line-height: 1.5;
    }

    .suggestion-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        max-width: 650px;
        margin: 0 auto;
    }

    .suggestion-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 14px 16px;
        text-align: left;
    }
    .suggestion-card:hover {
        background: rgba(0, 80, 255, 0.07);
        border-color: rgba(0, 80, 255, 0.25);
    }
    .suggestion-card .stitle {
        font-size: 13px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.85);
        margin-bottom: 4px;
    }
    .suggestion-card .sdesc {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.4);
    }

    /* Message Bubbles */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 14px 12px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 14px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: transparent !important;
        border-radius: 14px !important;
        margin-bottom: 12px !important;
    }

    /* Expander Styling (Source Chunks) — Opaque, No Bleed-Through */
    [data-testid="stExpander"] {
        background: #0d1124 !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 10px !important;
        margin-top: 8px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stExpander"] summary {
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 12px !important;
        padding: 8px 12px !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #00D6FF !important;
    }

    .source-chip {
        background: rgba(0, 80, 255, 0.06);
        border: 1px solid rgba(0, 80, 255, 0.18);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 12px;
        color: rgba(255, 255, 255, 0.75);
        line-height: 1.5;
    }
    .source-chip strong { color: #58a6ff; }
    .source-chip em { color: rgba(255, 255, 255, 0.45); font-style: normal; }

    /* ═══════════════════════════════════════════════════════════════════
       FIXED BOTTOM CHAT INPUT BAR — SOLID BACKGROUND, ZERO OVERLAP
       ═══════════════════════════════════════════════════════════════════ */
    [data-testid="stBottom"] {
        background: #060814 !important;
        border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-top: 12px !important;
        padding-bottom: 24px !important;
        z-index: 9999 !important;
    }
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }

    [data-testid="stChatInput"] {
        max-width: 840px !important;
        margin: 0 auto !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #0e1224 !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        font-size: 14px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #00D6FF !important;
        box-shadow: 0 0 16px rgba(0, 214, 255, 0.3) !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #0050FF, #00D6FF) !important;
        border: none !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Ambient Background HTML ───────────────────────────────────────────────────
st.markdown(
    """
    <div class="animated-bg">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    </div>
    <div class="grid-overlay"></div>
    """,
    unsafe_allow_html=True,
)


# ── Session State Initialization ───────────────────────────────────────────────
def get_qa_chain_for_model(model_display_name: str) -> QAChain:
    """Build a QAChain instance for the selected free model."""
    cfg = AVAILABLE_FREE_MODELS[model_display_name]
    provider = cfg["provider"]
    api_key = API_KEYS.get(provider, "")
    return QAChain(
        provider=provider,
        api_key=api_key,
        model_name=cfg["model"],
        temperature=0.2,
    )


def init_session_state():
    defaults = {
        "messages": [],
        "pdf_processor": PDFProcessor(chunk_size=500, chunk_overlap=100),
        "vector_store": VectorStoreManager(
            persist_directory=VECTOR_DIR,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        "selected_model_display": DEFAULT_MODEL,
        "qa_chain": None,
        "processed_files": set(),
        "doc_stats": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Initialize chain if not set
    if st.session_state.qa_chain is None:
        try:
            st.session_state.qa_chain = get_qa_chain_for_model(
                st.session_state.selected_model_display
            )
        except Exception:
            pass


init_session_state()
vs = st.session_state.vector_store
doc_count = vs.document_count
sources = vs.get_stored_sources()
is_small = vs.is_small_corpus() if doc_count > 0 else False
total_words = vs.get_total_words() if doc_count > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT SIDEBAR — CHATGPT-STYLE CONTROL CENTER (NO API KEYS REQUIRED)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-glow"></div>
            <span>RAG Document Q&A</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. New Chat Button
    if st.button("＋ New Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

    # 2. Free Model Switcher
    st.markdown('<div class="sidebar-section-title">🤖 Active Free Model</div>', unsafe_allow_html=True)

    model_keys = list(AVAILABLE_FREE_MODELS.keys())
    current_idx = (
        model_keys.index(st.session_state.selected_model_display)
        if st.session_state.selected_model_display in model_keys
        else 0
    )

    selected_model_display = st.selectbox(
        "Select Model",
        model_keys,
        index=current_idx,
        label_visibility="collapsed",
        key="model_switcher_select",
    )

    # If model selection changed, update QA chain
    if selected_model_display != st.session_state.selected_model_display or st.session_state.qa_chain is None:
        try:
            st.session_state.qa_chain = get_qa_chain_for_model(selected_model_display)
            st.session_state.selected_model_display = selected_model_display
        except Exception as e:
            st.error(f"Error loading model: {e}")

    # Active model info card
    curr_cfg = AVAILABLE_FREE_MODELS[st.session_state.selected_model_display]
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:12px; font-weight:600; color:#fff;">{curr_cfg['badge']}</span>
                <span style="font-size:10px; color:#22c55e; font-weight:600;">● Active</span>
            </div>
            <div style="font-size:11px; color:rgba(255,255,255,0.45); line-height:1.4;">
                {curr_cfg['description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Document Upload Section
    st.markdown('<div class="sidebar-section-title">📁 Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf", dir=UPLOAD_DIR
                    )
                    tmp.write(uploaded_file.getvalue())
                    tmp.close()
                    try:
                        processor = st.session_state.pdf_processor
                        chunks, file_hash, stats = processor.process_pdf(
                            tmp.name, source_name=uploaded_file.name
                        )
                        existing = vs.get_stored_hashes()
                        if file_hash in existing:
                            st.info(f"ℹ️ {uploaded_file.name} already indexed.")
                        else:
                            count = vs.add_documents(chunks)
                            st.success(f"Indexed {count} chunks ({stats['words']} words)")
                        st.session_state.processed_files.add(uploaded_file.name)
                        st.session_state.doc_stats[uploaded_file.name] = stats
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        os.unlink(tmp.name)
        doc_count = vs.document_count
        sources = vs.get_stored_sources()
        is_small = vs.is_small_corpus()
        total_words = vs.get_total_words()

    # Knowledge Base Stats Card
    if sources:
        mode_badge = "📖 Full-Context" if is_small else "🔍 RAG Retrieval"
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div style="font-size:12px; font-weight:600; color:#fff; margin-bottom:4px;">
                    Knowledge Base ({len(sources)} files)
                </div>
                <div style="font-size:11px; color:rgba(255,255,255,0.5);">
                    • {doc_count} chunks · ~{total_words:,} words<br/>
                    • Mode: <span style="color:#00D6FF;">{mode_badge}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Retrieval Settings
    st.markdown('<div class="sidebar-section-title">⚙️ Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K chunks", 1, 15, 5, help="Number of chunks retrieved for larger documents")
    source_filter = st.selectbox(
        "Filter by document",
        ["All documents"] + sources,
        index=0,
    )
    show_sources = st.checkbox("Show source citations", value=True)

    # 5. Clear & Reset Actions
    st.markdown('<div class="sidebar-section-title">🗑️ Actions</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    if col_a.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col_b.button("Reset KB", use_container_width=True):
        vs.clear_all()
        st.session_state.processed_files = set()
        st.session_state.doc_stats = {}
        st.session_state.messages = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — CLEAN CHATBOT INTERFACE (LIKE CHATGPT)
# ═══════════════════════════════════════════════════════════════════════════════

# Minimal Top Status Pill
curr_badge = curr_cfg["badge"]
doc_info = (
    f"{len(sources)} Document{'s' if len(sources) != 1 else ''} Ready"
    if sources
    else "No documents uploaded"
)

st.markdown(
    f"""
    <div class="chat-top-status">
        <div class="model-badge">
            <span class="status-dot-green"></span>
            <span>{curr_badge}</span>
        </div>
        <div style="font-size:12px; color:rgba(255,255,255,0.45);">
            📄 {doc_info}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Empty State Welcome Hero (ChatGPT style)
if not st.session_state.messages:
    st.markdown(
        """
        <div class="chat-empty-hero">
            <div class="chat-empty-icon">🤖</div>
            <div class="chat-empty-title">What would you like to know?</div>
            <div class="chat-empty-subtitle">
                Upload your PDF in the left sidebar, and ask anything. Answers are grounded in your document.
            </div>
            <div class="suggestion-grid">
                <div class="suggestion-card">
                    <div class="stitle">📄 Summarize Document</div>
                    <div class="sdesc">Give me a high-level summary of the entire document.</div>
                </div>
                <div class="suggestion-card">
                    <div class="stitle">🔍 Key Insights</div>
                    <div class="sdesc">Extract the most important points and findings.</div>
                </div>
                <div class="suggestion-card">
                    <div class="stitle">💡 Detailed Breakdown</div>
                    <div class="sdesc">Explain specific sections, numbers, or technical concepts.</div>
                </div>
                <div class="suggestion-card">
                    <div class="stitle">🎯 Actionable Takeaways</div>
                    <div class="sdesc">What are the practical next steps or conclusions?</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render Chat History
for msg in st.session_state.messages:
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and show_sources:
            with st.expander("📚 Source chunks", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-chip">'
                        f'<strong>{src["file"]}</strong> · Page {src["page"]} · '
                        f'Chunk {src["chunk"]}<br/>'
                        f'<em>{src["preview"]}</em></div>',
                        unsafe_allow_html=True,
                    )

# Pinned Chat Input at bottom
if prompt := st.chat_input("Ask a question about your documents..."):
    if doc_count == 0:
        st.warning("⚠️ Please upload at least one PDF in the left sidebar first.")
        st.stop()

    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing documents & generating answer..."):
            try:
                filter_src = None if source_filter == "All documents" else source_filter
                use_full_context = vs.is_small_corpus()

                if use_full_context:
                    if filter_src:
                        retrieved = vs.get_all_by_source(filter_src)
                    else:
                        retrieved = vs.get_all_documents()
                else:
                    retrieved = vs.similarity_search(
                        query=prompt, k=top_k, source_filter=filter_src
                    )

                if not retrieved:
                    answer_text = (
                        "I couldn't find relevant information in the uploaded "
                        "documents to answer your question. Try rephrasing or uploading more material."
                    )
                    sources_data = []
                    answer_mode = None
                else:
                    result = st.session_state.qa_chain.answer(
                        question=prompt,
                        retrieved_docs=retrieved,
                        full_context=use_full_context,
                    )
                    answer_text = result["answer"]
                    sources_data = result["sources"]
                    answer_mode = result["mode"]

                st.markdown(answer_text)

                if answer_mode:
                    mode_emoji = "📖" if answer_mode == "full-context" else "🔍"
                    mode_desc = (
                        f"Full document context analyzed ({result.get('model', 'llm')})"
                        if answer_mode == "full-context"
                        else f"Top-{len(sources_data)} chunks retrieved ({result.get('model', 'llm')})"
                    )
                    st.caption(f"{mode_emoji} *{mode_desc}*")

                if sources_data and show_sources:
                    with st.expander("📚 Source chunks", expanded=False):
                        for src in sources_data:
                            st.markdown(
                                f'<div class="source-chip">'
                                f'<strong>{src["file"]}</strong> · Page {src["page"]} · '
                                f'Chunk {src["chunk"]}<br/>'
                                f'<em>{src["preview"]}</em></div>',
                                unsafe_allow_html=True,
                            )

                msg_data = {"role": "assistant", "content": answer_text}
                if sources_data:
                    msg_data["sources"] = sources_data
                st.session_state.messages.append(msg_data)

            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
