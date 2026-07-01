import os
import sys
import streamlit as st

# =====================================
# Import Paths
# =====================================
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from loader import load_documents
from splitter import split_documents
from vectorstore import create_vectorstore
from rag_chain import create_rag_chain, ask_question

from agents.planner_agent import PlannerAgent
from agents.summarizer_agent import SummarizerAgent
from agents.comparison_agent import ComparisonAgent

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="DocPilot-AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# Custom Theme / CSS
# =====================================
st.markdown("""
<style>

    /* ---------- Global ---------- */
    .stApp {
        background-color: #0E1117;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* ============================================================
       SIDEBAR — this was the root cause of the white/invisible UI.
       Streamlit's sidebar is its own DOM node (data-testid="stSidebar")
       and does NOT inherit .stApp's background. Every rule below
       explicitly targets the sidebar so nothing falls back to the
       default light theme.
       ============================================================ */
    [data-testid="stSidebar"] {
        background-color: #0B0E14 !important;
        border-right: 1px solid #1F2530;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #0B0E14 !important;
    }

    /* Any plain text / labels / captions inside the sidebar */
    [data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #9CA3AF !important;
    }

    /* ---------- Sidebar Buttons ---------- */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(145deg, #1A2029, #161B22) !important;
        color: #F1F3F5 !important;
        border: 1px solid #2E3543 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.35);
        color: #FFFFFF !important;
    }

    /* ---------- File Uploader Dropzone ---------- */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #161B22 !important;
        border: 1px dashed #2E3543 !important;
        border-radius: 12px !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #B4BAC4 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: #1A2029 !important;
        color: #F1F3F5 !important;
        border: 1px solid #2E3543 !important;
    }

    /* Uploaded-file chip that appears after selecting a file */
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
        background: #161B22 !important;
        border-radius: 8px !important;
        color: #E5E7EB !important;
    }

    /* ---------- Alerts (st.success / st.warning / st.error / st.info) ---------- */
    [data-testid="stSidebar"] [data-testid="stAlert"],
    .stApp [data-testid="stAlert"] {
        background: #161B22 !important;
        border: 1px solid #262C36 !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] p,
    .stApp [data-testid="stAlert"] p {
        color: #F1F3F5 !important;
    }

    /* success = green accent border, warning = amber accent border */
    [data-testid="stSidebar"] div[data-baseweb="notification"][kind="success"],
    [data-testid="stSidebar"] .stSuccess {
        border-left: 3px solid #22C55E !important;
    }

    [data-testid="stSidebar"] .stWarning {
        border-left: 3px solid #F59E0B !important;
    }

    /* ---------- Expander ("Indexed files") ---------- */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #161B22 !important;
        border: 1px solid #262C36 !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #E5E7EB !important;
        background: #161B22 !important;
    }

    /* ---------- Divider ---------- */
    [data-testid="stSidebar"] hr {
        border-color: #262C36 !important;
    }

    /* ---------- Hero ---------- */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #8B5CF6 0%, #22D3EE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #B4BAC4;
        margin-bottom: 1.4rem;
    }

    /* ---------- Feature Cards ---------- */
    .feature-card {
        background: linear-gradient(145deg, #161B22, #1A2029);
        border: 1px solid #262C36;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
    }

    .feature-card h4 {
        margin: 0 0 0.3rem 0;
        color: #E5E7EB;
        font-size: 1rem;
    }

    .feature-card p {
        margin: 0;
        color: #B4BAC4;
        font-size: 0.85rem;
    }

    /* ---------- Sidebar Branding ---------- */
    .sidebar-brand {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #8B5CF6 0%, #22D3EE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .sidebar-tag {
        color: #6B7280 !important;
        font-size: 0.78rem;
        margin-top: -6px;
        margin-bottom: 1rem;
    }

    /* ---------- KB Stats ---------- */
    .kb-stat-box {
        background: #161B22;
        border: 1px solid #262C36;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        text-align: center;
    }

    .kb-stat-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #22D3EE !important;
    }

    .kb-stat-label {
        font-size: 0.72rem;
        color: #9CA3AF !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* ---------- Agent Status Panel ---------- */
    .agent-pill {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: #161B22;
        border: 1px solid #262C36;
        border-radius: 10px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.45rem;
        font-size: 0.85rem;
        color: #9CA3AF !important;
        transition: all 0.2s ease;
    }

    .agent-pill.active {
        border: 1px solid #8B5CF6;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.35);
        color: #E5E7EB !important;
        background: linear-gradient(145deg, #1E1730, #161B22);
    }

    .agent-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4B5563;
    }

    .agent-pill.active .agent-dot {
        background: #22D3EE;
        box-shadow: 0 0 8px #22D3EE;
    }

    /* ---------- Chat Agent Badge ---------- */
    .agent-badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        margin-bottom: 0.5rem;
        background: rgba(139, 92, 246, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }

    /* ---------- Readability Fixes ---------- */
    /* Streamlit's default markdown text renders too dim on dark backgrounds.
       Force high-contrast text everywhere answers/content actually render. */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #F1F3F5 !important;
        line-height: 1.65 !important;
        font-size: 0.98rem !important;
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #FFFFFF !important;
        margin-top: 0.9rem !important;
        margin-bottom: 0.3rem !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #FFFFFF !important;
    }

    /* Give chat bubbles a proper card surface instead of floating on bare black */
    [data-testid="stChatMessage"] {
        background: #161B22 !important;
        border: 1px solid #262C36 !important;
        border-radius: 14px !important;
        padding: 1rem 1.1rem !important;
        margin-bottom: 0.7rem !important;
    }

    /* Slightly distinguish the user's own messages */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #1A1F2B !important;
        border: 1px solid #2E3543 !important;
    }

    /* ---------- Chat Input Box ---------- */
    [data-testid="stChatInput"] {
        background: #161B22 !important;
        border: 1px solid #262C36 !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #F1F3F5 !important;
    }

</style>
""", unsafe_allow_html=True)

# =====================================
# Create Required Folders
# =====================================
os.makedirs("data", exist_ok=True)
os.makedirs("faiss_db", exist_ok=True)

# =====================================
# Session State
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "llm" not in st.session_state:
    st.session_state.llm = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "last_agent" not in st.session_state:
    st.session_state.last_agent = None

planner = PlannerAgent()
summarizer = SummarizerAgent()
comparison = ComparisonAgent()

AGENT_META = {
    "retrieve":  {"label": "Retrieval Agent",   "icon": "🔎"},
    "summarize": {"label": "Summarizer Agent",  "icon": "📝"},
    "compare":   {"label": "Comparison Agent",  "icon": "📊"},
    "quiz":      {"label": "Quiz Agent",        "icon": "❓"},
}

# =====================================
# Sidebar
# =====================================
with st.sidebar:

    st.markdown('<p class="sidebar-brand">🧭 DocPilot-AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-tag">Multi-Agent Document Research Assistant</p>', unsafe_allow_html=True)

    st.markdown("#### 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    process_button = st.button("🚀 Process Documents", use_container_width=True)

    st.divider()

    # ---------- Knowledge Base Stats ----------
    st.markdown("#### 📊 Knowledge Base")

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.markdown(f"""
        <div class="kb-stat-box">
            <div class="kb-stat-value">{len(st.session_state.doc_names)}</div>
            <div class="kb-stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col2:
        st.markdown(f"""
        <div class="kb-stat-box">
            <div class="kb-stat-value">{st.session_state.chunk_count}</div>
            <div class="kb-stat-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    if st.session_state.rag_ready:
        st.success("✅ Knowledge base ready")
    else:
        st.warning("⚠️ No documents indexed yet")

    if st.session_state.doc_names:
        with st.expander("📁 Indexed files"):
            for name in st.session_state.doc_names:
                st.caption(f"• {name}")

    st.divider()

    # ---------- Agent Status Panel ----------
    st.markdown("#### 🧠 Agent Status")

    for key, meta in AGENT_META.items():
        active_class = "active" if st.session_state.last_agent == key else ""
        st.markdown(f"""
        <div class="agent-pill {active_class}">
            <div class="agent-dot"></div>
            <div>{meta['icon']} {meta['label']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_agent = None
        st.rerun()

# =====================================
# Main Page — Hero
# =====================================
st.markdown('<p class="hero-title">DocPilot-AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">A multi-agent AI system that plans, retrieves, summarizes, '
    'and compares information across your documents.</p>',
    unsafe_allow_html=True
)

feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)

with feat_col1:
    st.markdown("""
    <div class="feature-card">
        <h4>🔎 Retrieve</h4>
        <p>Grounded, semantic Q&A over your documents.</p>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="feature-card">
        <h4>📝 Summarize</h4>
        <p>Structured, executive-style document summaries.</p>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Compare</h4>
        <p>Cross-document analysis and comparison tables.</p>
    </div>
    """, unsafe_allow_html=True)

with feat_col4:
    st.markdown("""
    <div class="feature-card">
        <h4>❓ Quiz</h4>
        <p>Auto-generated quiz questions. (Coming soon)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =====================================
# Process Documents
# =====================================
if process_button:

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

    else:

        for uploaded_file in uploaded_files:

            file_path = os.path.join("data", uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("🔄 Extracting text, chunking, and building the FAISS index..."):

            docs = load_documents()

            chunks = split_documents(docs)

            db = create_vectorstore(chunks)

            llm, retriever = create_rag_chain(db)

            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.rag_ready = True
            st.session_state.doc_names = [f.name for f in uploaded_files]
            st.session_state.chunk_count = len(chunks)

        st.success(f"✅ Knowledge base built — {len(chunks)} chunks indexed from {len(uploaded_files)} document(s).")
        st.rerun()

# =====================================
# Chat History
# =====================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "assistant" and message.get("agent"):
            meta = AGENT_META.get(message["agent"])
            if meta:
                st.markdown(
                    f'<span class="agent-badge">{meta["icon"]} {meta["label"]}</span>',
                    unsafe_allow_html=True
                )

        st.markdown(message["content"])

# =====================================
# Chat Input
# =====================================
user_question = st.chat_input(
    "Ask DocPilot anything about your uploaded documents..."
)

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    agent_used = None

    if not st.session_state.rag_ready:

        reply = "⚠️ Please upload and process your documents first — I don't have a knowledge base to work with yet."

    else:

        with st.spinner("🧠 Planner is choosing the right agent..."):
            task = planner.plan(user_question)
            agent_used = task if task in AGENT_META else "retrieve"

        st.session_state.last_agent = agent_used

        agent_label = AGENT_META.get(agent_used, AGENT_META["retrieve"])["label"]

        with st.spinner(f"⚙️ {agent_label} is working on your request..."):

            if task == "retrieve":

                reply = ask_question(
                    st.session_state.llm,
                    st.session_state.retriever,
                    user_question
                )

            elif task == "summarize":

                reply = summarizer.summarize(
                    st.session_state.llm,
                    st.session_state.retriever
                )

            elif task == "compare":

                reply = comparison.compare(
                    st.session_state.llm,
                    st.session_state.retriever
                )

            elif task == "quiz":

                reply = "🚧 Quiz Agent is coming soon."

            else:

                reply = ask_question(
                    st.session_state.llm,
                    st.session_state.retriever,
                    user_question
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "agent": agent_used
        }
    )

    with st.chat_message("assistant"):
        if agent_used and agent_used in AGENT_META:
            meta = AGENT_META[agent_used]
            st.markdown(
                f'<span class="agent-badge">{meta["icon"]} {meta["label"]}</span>',
                unsafe_allow_html=True
            )
        st.markdown(reply)
