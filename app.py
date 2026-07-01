"""
DocPilot-AI — Streamlit entrypoint.

This module wires the Streamlit UI to the multi-agent backend:

    User -> Planner Agent (LLM intent classification)
              -> Retrieval Agent   (retrieve / explain / search)
              -> Summarizer Agent  (summarize)
              -> Comparison Agent  (compare)
              -> Quiz Agent        (quiz)

The UI, theme, and layout are unchanged from the original app.py.
Only the orchestration logic and response rendering were updated to
support the new structured (dict-based) agent outputs and real
multi-agent routing.
"""

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

# QuizAgent and ConversationMemory are new components introduced by the
# multi-agent upgrade. They are imported defensively so this app.py keeps
# working even before those files exist / are upgraded.
try:
    from agents.quiz_agent import QuizAgent
except ImportError:
    QuizAgent = None

try:
    from memory import ConversationMemory
except ImportError:
    ConversationMemory = None

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
    /* The chat input is a multi-layer widget (outer container > baseweb
       wrapper > textarea). Each layer ships its own white background,
       so every layer must be overridden explicitly or it falls back
       to white-on-white like the sidebar did. */
    [data-testid="stChatInput"] {
        background: #161B22 !important;
        border: 1px solid #262C36 !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background: #161B22 !important;
        border: none !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] {
        background: #161B22 !important;
        color: #F1F3F5 !important;
        caret-color: #F1F3F5 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #7C8493 !important;
        opacity: 1 !important;
    }

    /* Send button */
    [data-testid="stChatInput"] button {
        background: #EF4444 !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: #FFFFFF !important;
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

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory() if ConversationMemory else None

if "quiz_difficulty" not in st.session_state:
    st.session_state.quiz_difficulty = "Medium"

if "quiz_num_questions" not in st.session_state:
    st.session_state.quiz_num_questions = 5

planner = PlannerAgent()
summarizer = SummarizerAgent()
comparison = ComparisonAgent()
quiz_agent = QuizAgent() if QuizAgent else None

# Metadata used purely for sidebar/status/badge rendering.
# Keys correspond to the *planner intents*, not necessarily distinct
# backend agents — e.g. "explain" and "search" are routed to the
# Retrieval Agent under the hood but are surfaced with their own label
# so the user can see what the Planner actually decided.
AGENT_META = {
    "retrieve":  {"label": "Retrieval Agent",   "icon": "🔎"},
    "explain":   {"label": "Explain (Retrieval)", "icon": "💡"},
    "search":    {"label": "Search (Retrieval)", "icon": "🔍"},
    "summarize": {"label": "Summarizer Agent",  "icon": "📝"},
    "compare":   {"label": "Comparison Agent",  "icon": "📊"},
    "quiz":      {"label": "Quiz Agent",        "icon": "❓"},
}

# Maps every planner intent onto the backend handler responsible for it.
INTENT_TO_HANDLER = {
    "retrieve": "retrieve",
    "explain": "retrieve",
    "search": "retrieve",
    "summarize": "summarize",
    "compare": "compare",
    "quiz": "quiz",
}


# =====================================
# Response Formatting Helpers
# =====================================
def _get_recent_chat_history(max_turns: int = 6):
    """
    Build a lightweight chat history list from session messages for
    agents that support conversational follow-ups (Planner, Retrieval).

    Returns a list of {"role": ..., "content": ...} dicts, most recent
    `max_turns` exchanges only, to keep prompts small.
    """
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    return history[-max_turns:]


def format_retrieve_response(data) -> str:
    """Render a Retrieval Agent response (dict or legacy string) as markdown."""
    if isinstance(data, str):
        return data

    if not isinstance(data, dict):
        return str(data)

    parts = []

    answer = data.get("answer", "").strip()
    if answer:
        parts.append(answer)

    key_points = data.get("key_points")
    if key_points:
        parts.append("**Key Points:**")
        parts.append("\n".join(f"- {kp}" for kp in key_points))

    confidence = data.get("confidence")
    if confidence:
        parts.append(f"**Confidence:** {confidence}")

    sources = data.get("sources")
    if sources:
        lines = ["**Sources:**"]
        for src in sources:
            doc_name = src.get("document", "Unknown document")
            chunk_count = src.get("chunk_count")
            pages = src.get("pages")

            line = f"- 📄 **{doc_name}**"
            details = []
            if pages:
                details.append(f"pages: {', '.join(str(p) for p in pages)}")
            if chunk_count is not None:
                details.append(f"{chunk_count} chunk(s) used")
            if details:
                line += f" _({'; '.join(details)})_"
            lines.append(line)
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "I couldn't find an answer in the documents."


def format_summarize_response(data) -> str:
    """Render a Summarizer Agent response (dict or legacy string) as markdown."""
    if isinstance(data, str):
        return data

    if not isinstance(data, dict):
        return str(data)

    sections = [
        ("📌 Executive Summary", data.get("executive_summary")),
        ("🔑 Key Findings", data.get("key_findings")),
        ("💡 Important Concepts", data.get("important_concepts")),
        ("✅ Action Items", data.get("action_items")),
        ("🏁 Conclusion", data.get("conclusion")),
    ]

    parts = []
    for title, content in sections:
        if not content:
            continue
        parts.append(f"**{title}**")
        if isinstance(content, list):
            parts.append("\n".join(f"- {item}" for item in content))
        else:
            parts.append(str(content))

    return "\n\n".join(parts) if parts else "No summary could be generated."


def format_compare_response(data) -> str:
    """Render a Comparison Agent response (dict or legacy string) as markdown."""
    if isinstance(data, str):
        return data

    if not isinstance(data, dict):
        return str(data)

    if data.get("single_document"):
        note = data.get(
            "message",
            "Only one document is indexed, so a cross-document comparison "
            "isn't possible. Upload at least two PDFs to compare them."
        )
        return f"⚠️ {note}"

    parts = []

    table = data.get("comparison_table")
    if table:
        # Accept either a pre-formatted markdown table string or a list
        # of row dicts to render as a markdown table.
        if isinstance(table, str):
            parts.append("**📊 Comparison Table**\n\n" + table)
        elif isinstance(table, list) and table:
            headers = list(table[0].keys())
            header_row = "| " + " | ".join(headers) + " |"
            divider_row = "| " + " | ".join("---" for _ in headers) + " |"
            body_rows = [
                "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
                for row in table
            ]
            md_table = "\n".join([header_row, divider_row, *body_rows])
            parts.append("**📊 Comparison Table**\n\n" + md_table)

    list_sections = [
        ("🤝 Similarities", data.get("similarities")),
        ("⚡ Differences", data.get("differences")),
        ("💪 Strengths", data.get("strengths")),
        ("⚠️ Weaknesses", data.get("weaknesses")),
        ("🧭 Recommendations", data.get("recommendations")),
    ]

    for title, content in list_sections:
        if not content:
            continue
        parts.append(f"**{title}**")
        if isinstance(content, list):
            parts.append("\n".join(f"- {item}" for item in content))
        else:
            parts.append(str(content))

    return "\n\n".join(parts) if parts else "No comparison could be generated."


def format_quiz_response(data) -> str:
    """Render a Quiz Agent response (dict or legacy string) as markdown."""
    if isinstance(data, str):
        return data

    if not isinstance(data, dict):
        return str(data)

    parts = []
    difficulty = data.get("difficulty")
    if difficulty:
        parts.append(f"**Difficulty:** {difficulty}")

    mcqs = data.get("mcqs")
    if mcqs:
        lines = ["**📝 Multiple Choice Questions**"]
        for i, q in enumerate(mcqs, start=1):
            lines.append(f"\n{i}. {q.get('question', '')}")
            for opt_letter, opt_text in q.get("options", {}).items():
                lines.append(f"   - {opt_letter}) {opt_text}")
            answer = q.get("answer")
            if answer:
                lines.append(f"   - ✅ **Answer:** {answer}")
        parts.append("\n".join(lines))

    true_false = data.get("true_false")
    if true_false:
        lines = ["**✅❌ True / False**"]
        for i, q in enumerate(true_false, start=1):
            lines.append(f"{i}. {q.get('statement', '')}")
            answer = q.get("answer")
            if answer is not None:
                lines.append(f"   - **Answer:** {answer}")
        parts.append("\n".join(lines))

    short_answer = data.get("short_answer")
    if short_answer:
        lines = ["**✍️ Short Answer**"]
        for i, q in enumerate(short_answer, start=1):
            lines.append(f"{i}. {q.get('question', '')}")
            answer = q.get("answer")
            if answer:
                lines.append(f"   - **Suggested Answer:** {answer}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "No quiz questions could be generated."


FORMATTERS = {
    "retrieve": format_retrieve_response,
    "explain": format_retrieve_response,
    "search": format_retrieve_response,
    "summarize": format_summarize_response,
    "compare": format_compare_response,
    "quiz": format_quiz_response,
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

    # ---------- Quiz Settings ----------
    st.markdown("#### ❓ Quiz Settings")

    st.session_state.quiz_difficulty = st.selectbox(
        "Difficulty",
        options=["Easy", "Medium", "Hard"],
        index=["Easy", "Medium", "Hard"].index(st.session_state.quiz_difficulty),
    )

    st.session_state.quiz_num_questions = st.slider(
        "Questions per quiz",
        min_value=3,
        max_value=10,
        value=st.session_state.quiz_num_questions,
    )

    st.caption("Ask things like *\"quiz me on this document\"* to trigger the Quiz Agent.")

    st.divider()

    # ---------- Agent Status Panel ----------
    st.markdown("#### 🧠 Agent Status")

    for key in ["retrieve", "summarize", "compare", "quiz"]:
        meta = AGENT_META[key]
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
        if st.session_state.memory and hasattr(st.session_state.memory, "clear"):
            st.session_state.memory.clear()
        st.rerun()

# =====================================
# Main Page — Hero
# =====================================
st.markdown('<p class="hero-title">DocPilot-AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">A multi-agent AI system that plans, retrieves, summarizes, '
    'compares, and quizzes you on your documents.</p>',
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
        <p>Auto-generated MCQs, true/false, and short answer questions.</p>
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

            # Reset conversational memory whenever the knowledge base changes,
            # since prior follow-up context no longer applies to a new corpus.
            if ConversationMemory:
                st.session_state.memory = ConversationMemory()

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

        chat_history = _get_recent_chat_history()

        with st.spinner("🧠 Planner is reasoning about the right agent..."):
            # PlannerAgent.plan is expected to use the LLM to classify intent
            # into one of: retrieve, summarize, compare, quiz, explain, search.
            try:
                intent = planner.plan(user_question, chat_history=chat_history)
            except TypeError:
                # Backward-compatible call for a planner that doesn't yet
                # accept chat_history.
                intent = planner.plan(user_question)

        task = INTENT_TO_HANDLER.get(intent, "retrieve")
        agent_used = intent if intent in AGENT_META else "retrieve"

        st.session_state.last_agent = task

        agent_label = AGENT_META.get(agent_used, AGENT_META["retrieve"])["label"]

        with st.spinner(f"⚙️ {agent_label} is working on your request..."):

            if task == "retrieve":

                try:
                    raw_reply = ask_question(
                        st.session_state.llm,
                        st.session_state.retriever,
                        user_question,
                        chat_history=chat_history,
                    )
                except TypeError:
                    raw_reply = ask_question(
                        st.session_state.llm,
                        st.session_state.retriever,
                        user_question,
                    )

                reply = format_retrieve_response(raw_reply)

            elif task == "summarize":

                try:
                    raw_reply = summarizer.summarize(
                        st.session_state.llm,
                        st.session_state.retriever,
                        doc_names=st.session_state.doc_names,
                    )
                except TypeError:
                    raw_reply = summarizer.summarize(
                        st.session_state.llm,
                        st.session_state.retriever,
                    )

                reply = format_summarize_response(raw_reply)

            elif task == "compare":

                try:
                    raw_reply = comparison.compare(
                        st.session_state.llm,
                        st.session_state.retriever,
                        doc_names=st.session_state.doc_names,
                    )
                except TypeError:
                    raw_reply = comparison.compare(
                        st.session_state.llm,
                        st.session_state.retriever,
                    )

                reply = format_compare_response(raw_reply)

            elif task == "quiz":

                if quiz_agent is None:
                    reply = "🚧 Quiz Agent is not available yet — add `agents/quiz_agent.py` to enable it."
                else:
                    raw_reply = quiz_agent.generate_quiz(
                        st.session_state.llm,
                        st.session_state.retriever,
                        difficulty=st.session_state.quiz_difficulty,
                        num_questions=st.session_state.quiz_num_questions,
                    )
                    reply = format_quiz_response(raw_reply)

            else:

                raw_reply = ask_question(
                    st.session_state.llm,
                    st.session_state.retriever,
                    user_question,
                )
                reply = format_retrieve_response(raw_reply)

        # Persist the exchange in conversational memory, if available, so
        # follow-up questions can be answered with context.
        if st.session_state.memory and hasattr(st.session_state.memory, "add_exchange"):
            st.session_state.memory.add_exchange(user_question, reply)

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
