"""
DocPilot-AI Streamlit entrypoint.

This module wires the Streamlit UI to the multi-agent backend:
    User -> Planner Agent (LLM intent classification)
              -> Retrieval Agent   (retrieve / explain / search)
              -> Summarizer Agent  (summarize)
              -> Comparison Agent  (compare)
              -> Quiz Agent        (quiz)
"""

import os
import streamlit as st
from langchain_groq import ChatGroq
from groq import Groq

from utils.loader import load_documents
from utils.splitter import split_documents
from utils.vectorstore import create_vectorstore
from utils.rag_chain import create_rag_chain, _get_api_key

from agents import (
    PlannerAgent,
    RetrievalAgent,
    SummarizerAgent,
    ComparisonAgent,
    QuizAgent,
)
from utils.memory import ConversationMemory

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="DocPilot-AI",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# Custom Theme / CSS
# =====================================
st.markdown("""
<style>
    /* Global */
    .stApp {
        background-color: #0E1117;
    }
    #MainMenu, footer, [data-testid="stHeaderActionElements"] {
        visibility: hidden;
    }
    header {
        background-color: transparent !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stHeaderCollapsedControl"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B0E14 !important;
        border-right: 1px solid #1F2530;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #0B0E14 !important;
    }
    [data-testid="stSidebar"] * {
        color: #E5E7EB;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #9CA3AF !important;
    }
    /* Buttons */
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
    /* Dropzone */
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
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
        background: #161B22 !important;
        border-radius: 8px !important;
        color: #E5E7EB !important;
    }
    /* Alerts */
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
    [data-testid="stSidebar"] div[data-baseweb="notification"][kind="success"],
    [data-testid="stSidebar"] .stSuccess {
        border-left: 3px solid #22C55E !important;
    }
    [data-testid="stSidebar"] .stWarning {
        border-left: 3px solid #F59E0B !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #161B22 !important;
        border: 1px solid #262C36 !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #E5E7EB !important;
        background: #161B22 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #262C36 !important;
    }
    /* Hero */
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
    /* Feature Cards */
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
    .agent-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 8px;
        background: #161B22;
        border: 1px solid #262C36;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .agent-pill.active {
        border-color: #8B5CF6;
        background: rgba(139, 92, 246, 0.1);
    }
    .agent-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #6B7280;
    }
    .agent-pill.active .agent-dot {
        background-color: #22C55E;
        box-shadow: 0 0 6px #22C55E;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.35);
        color: #C4B5FD !important;
        font-size: 0.75rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# Directories & Credentials
# =====================================
os.makedirs("data", exist_ok=True)
os.makedirs("faiss_db", exist_ok=True)

api_key = _get_api_key()
if not api_key:
    st.sidebar.error("⚠️ `GROQ_API_KEY` is not found. Please set it in your `.env` file or Streamlit secrets.")

# =====================================
# Dynamic Live Model Fetcher
# =====================================
@st.cache_data(ttl=3600)
def get_available_groq_models(key: str):
    """Fetch active chat models directly from Groq API to avoid decommission errors."""
    fallback_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]
    if not key:
        return fallback_models
    try:
        client = Groq(api_key=key)
        model_list = client.models.list()
        chat_models = [
            m.id for m in model_list.data 
            if not m.id.startswith("whisper") and "safeguard" not in m.id
        ]
        return chat_models if chat_models else fallback_models
    except Exception:
        return fallback_models

# =====================================
# Session State Initialization
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

summarizer = SummarizerAgent()
comparison = ComparisonAgent()
quiz_agent = QuizAgent() if QuizAgent else None

AGENT_META = {
    "retrieve":  {"label": "Retrieval Agent",     "icon": "🔍"},
    "explain":   {"label": "Explain (Retrieval)", "icon": "💡"},
    "search":    {"label": "Search (Retrieval)",  "icon": "🔎"},
    "summarize": {"label": "Summarizer Agent",    "icon": "📝"},
    "compare":   {"label": "Comparison Agent",    "icon": "⚖️"},
    "quiz":      {"label": "Quiz Agent",          "icon": "❓"},
}

INTENT_TO_HANDLER = {
    "retrieve": "retrieve",
    "explain": "retrieve",
    "search": "retrieve",
    "summarize": "summarize",
    "compare": "compare",
    "quiz": "quiz",
}

# =====================================
# Formatters
# =====================================
def _get_recent_chat_history(max_turns: int = 6):
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    return history[-max_turns:]


def format_retrieve_response(data) -> str:
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
        parts.append("**Key Points:**\n" + "\n".join(f"- {kp}" for kp in key_points))
    confidence = data.get("confidence")
    if confidence:
        parts.append(f"**Confidence:** {confidence}")
    return "\n\n".join(parts) if parts else "I couldn't find an answer in the documents."


def format_summarize_response(data) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return str(data)
    sections = [
        ("Executive Summary", data.get("executive_summary")),
        ("Key Findings", data.get("key_findings")),
        ("Important Concepts", data.get("important_concepts")),
        ("Action Items", data.get("action_items")),
        ("Conclusion", data.get("conclusion")),
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
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return str(data)
    if data.get("single_document"):
        return data.get("message", "Please upload at least two PDFs to compare.")
    return str(data.get("comparison", data))


def format_quiz_response(data) -> str:
    if isinstance(data, str):
        return data
    return str(data)

# =====================================
# Sidebar UI
# =====================================
with st.sidebar:
    st.markdown('<p class="sidebar-brand">DocPilot-AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-tag">Multi-Agent Document Research Assistant</p>', unsafe_allow_html=True)

    st.markdown("#### Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    process_button = st.button("Process Documents", use_container_width=True)

    st.divider()

    st.markdown("#### LLM Model Selector")
    live_models = get_available_groq_models(api_key)
    selected_model = st.selectbox(
        "Choose Groq LLM",
        options=live_models,
        index=0,
        help="Real-time verified active models for your Groq API key."
    )

    st.divider()

    st.markdown("#### Knowledge Base")
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
        st.success("Knowledge base ready")
    else:
        st.warning("No documents indexed yet")

    if st.session_state.doc_names:
        with st.expander("Indexed files"):
            for name in st.session_state.doc_names:
                st.caption(name)

    st.divider()

    st.markdown("#### Quiz Settings")
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

    st.divider()

    st.markdown("#### Agent Status")
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

    if st.session_state.messages:
        export_text = "# DocPilot-AI Chat History\n\n"
        for msg in st.session_state.messages:
            role = msg["role"].upper()
            content = msg["content"]
            agent = f" (Agent: {msg.get('agent')})" if msg.get("agent") else ""
            export_text += f"### **{role}**{agent}\n{content}\n\n---\n\n"

        st.download_button(
            label="Download Chat (Markdown)",
            data=export_text,
            file_name="docpilot_chat_history.md",
            mime="text/markdown",
            use_container_width=True
        )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_agent = None
        if st.session_state.memory and hasattr(st.session_state.memory, "clear"):
            st.session_state.memory.clear()
        st.rerun()

# =====================================
# Main Header
# =====================================
st.markdown('<p class="hero-title">DocPilot-AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">A multi-agent AI system that plans, retrieves, summarizes, '
    'compares, and quizzes you on your documents.</p>',
    unsafe_allow_html=True
)

feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
with feat_col1:
    st.markdown('<div class="feature-card"><h4>Retrieve</h4><p>Grounded semantic Q&A.</p></div>', unsafe_allow_html=True)
with feat_col2:
    st.markdown('<div class="feature-card"><h4>Summarize</h4><p>Structured document summaries.</p></div>', unsafe_allow_html=True)
with feat_col3:
    st.markdown('<div class="feature-card"><h4>Compare</h4><p>Cross-document comparisons.</p></div>', unsafe_allow_html=True)
with feat_col4:
    st.markdown('<div class="feature-card"><h4>Quiz</h4><p>Auto-generated MCQs & questions.</p></div>', unsafe_allow_html=True)

st.markdown("")

# =====================================
# Document Processing
# =====================================
if process_button:
    if not api_key:
        st.error("Please configure your `GROQ_API_KEY` before processing documents.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        for filename in os.listdir("data"):
            file_to_remove = os.path.join("data", filename)
            if os.path.isfile(file_to_remove):
                os.remove(file_to_remove)

        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("Extracting text, chunking, and building the FAISS index..."):
            docs = load_documents()
            has_text = any(doc.page_content.strip() for doc in docs) if docs else False

            if not docs or not has_text:
                st.error("⚠️ No readable text found. Please upload a PDF with selectable text.")
            else:
                chunks = split_documents(docs)
                if not chunks:
                    st.error("⚠️ Failed to produce chunks from the uploaded document text.")
                else:
                    db = create_vectorstore(chunks)
                    llm, retriever = create_rag_chain(db, model_name=selected_model)

                    st.session_state.llm = llm
                    st.session_state.retriever = retriever
                    st.session_state.rag_ready = True
                    st.session_state.doc_names = [f.name for f in uploaded_files]
                    st.session_state.chunk_count = len(chunks)

                    if ConversationMemory:
                        st.session_state.memory = ConversationMemory()

                    st.success(f"Knowledge base built: {len(chunks)} chunks indexed from {len(uploaded_files)} document(s).")
                    st.rerun()

# =====================================
# Chat Messages
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
# Chat Handling
# =====================================
user_question = st.chat_input("Ask DocPilot anything about your uploaded documents...")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    agent_used = None

    if not api_key:
        reply = "⚠️ `GROQ_API_KEY` is missing. Please add it to your environment or secrets."
    elif not st.session_state.rag_ready:
        reply = "Please upload and process your documents first. I don't have a knowledge base to work with yet."
    else:
        chat_history = _get_recent_chat_history()

        active_llm = ChatGroq(
            api_key=api_key,
            model_name=selected_model,
            temperature=0
        )
        st.session_state.llm = active_llm

        lower_q = user_question.lower().strip()
        if "summarize" in lower_q or "summary" in lower_q:
            intent = "summarize"
        elif "compare" in lower_q or "comparison" in lower_q:
            intent = "compare"
        elif "quiz" in lower_q or "test me" in lower_q:
            intent = "quiz"
        else:
            with st.spinner("Planner is reasoning about the right agent..."):
                try:
                    planner = PlannerAgent(model_name=selected_model)
                    intent = planner.plan(user_question, chat_history=chat_history)
                except Exception:
                    intent = "retrieve"

        task = INTENT_TO_HANDLER.get(intent, "retrieve")
        agent_used = intent if intent in AGENT_META else "retrieve"
        st.session_state.last_agent = task

        agent_label = AGENT_META.get(agent_used, AGENT_META["retrieve"])["label"]

        try:
            with st.spinner(f"{agent_label} is working on your request..."):
                if task == "summarize":
                    raw_reply = summarizer.summarize(
                        st.session_state.llm,
                        st.session_state.retriever,
                        doc_names=st.session_state.doc_names,
                    )
                    reply = format_summarize_response(raw_reply)

                elif task == "compare":
                    raw_reply = comparison.compare(
                        st.session_state.llm,
                        st.session_state.retriever,
                        doc_names=st.session_state.doc_names,
                    )
                    reply = format_compare_response(raw_reply)

                elif task == "quiz":
                    if quiz_agent is None:
                        reply = "Quiz Agent is not available yet."
                    else:
                        raw_reply = quiz_agent.generate_quiz(
                            st.session_state.llm,
                            st.session_state.retriever,
                            difficulty=st.session_state.quiz_difficulty,
                            num_questions=st.session_state.quiz_num_questions,
                        )
                        reply = format_quiz_response(raw_reply)

                else:
                    retrieval_agent = RetrievalAgent()
                    raw_reply = retrieval_agent.retrieve(
                        st.session_state.llm,
                        st.session_state.retriever,
                        user_question,
                        chat_history=chat_history,
                    )
                    reply = format_retrieve_response(raw_reply)

        except Exception as e:
            reply = f"⚠️ Groq API Error with `{selected_model}`: {str(e)}\n\n*Tip: Try selecting another model from the left sidebar selector.*"

        if st.session_state.memory and hasattr(st.session_state.memory, "add_exchange"):
            st.session_state.memory.add_exchange(user_question, reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "agent": agent_used
    })

    with st.chat_message("assistant"):
        if agent_used and agent_used in AGENT_META:
            meta = AGENT_META[agent_used]
            st.markdown(
                f'<span class="agent-badge">{meta["icon"]} {meta["label"]}</span>',
                unsafe_allow_html=True
            )
        st.markdown(reply)
