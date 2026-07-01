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

# Planner Agent
from agents.planner_agent import PlannerAgent
from agents.summarizer_agent import SummarizerAgent
# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="🚀 DocPilot-AI",
    page_icon="🚀",
    layout="wide"
)

# =====================================
# Create Required Folders
# =====================================
os.makedirs("data", exist_ok=True)
os.makedirs("faiss_db", exist_ok=True)

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

# Initialize Planner Agent
planner = PlannerAgent()
summarizer = SummarizerAgent()
# =====================================
# Sidebar
# =====================================
with st.sidebar:

    st.title("🚀 DocPilot-AI")

    st.markdown("### 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "🚀 Process Documents",
        use_container_width=True
    )

    st.divider()

    st.markdown("### 🧠 Agent Skills")

    st.success("✅ Question Answering")

    st.info("📝 Document Summarization")

    st.info("📊 Document Comparison")

    st.info("❓ Quiz Generation")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =====================================
# Main Page
# =====================================
st.title("🚀 DocPilot-AI")

st.caption(
    "An Intelligent AI Research Agent for Multi-Document Analysis"
)

st.write(
    """
Upload one or multiple PDF documents and let DocPilot intelligently:

- 📚 Answer Questions
- 📝 Summarize Documents
- 📊 Compare Information
- ❓ Generate Quiz Questions
"""
)

# =====================================
# Process Documents
# =====================================
if process_button:

    if not uploaded_files:

        st.warning("Please upload at least one PDF.")

    else:

        for uploaded_file in uploaded_files:

            file_path = os.path.join(
                "data",
                uploaded_file.name
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("🔄 Building Knowledge Base..."):

            docs = load_documents()

            chunks = split_documents(docs)

            db = create_vectorstore(chunks)

            llm, retriever = create_rag_chain(db)

            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.rag_ready = True

        st.success(f"✅ {len(chunks)} chunks indexed successfully!")

# =====================================
# Display Chat History
# =====================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

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

    if not st.session_state.rag_ready:

        reply = "⚠️ Please upload and process your documents first."

    else:

        with st.spinner("🧠 DocPilot is planning the best response..."):

            task = planner.plan(user_question)

            if task == "retrieve":

                reply = ask_question(
                    st.session_state.llm,
                    st.session_state.retriever,
                    user_question
                )

            elif task == "summarize":

                reply = summarizer.summarize(
                        st.session_state.llm,
                        st.session_state.retriever,
                        "Provide a concise summary of the uploaded documents."
                    )
                

            elif task == "compare":

                reply = (
                    "🚧 **Comparison Agent** is coming in the next update."
                )

            elif task == "quiz":

                reply = (
                    "🚧 **Quiz Agent** is coming in the next update."
                )

            else:

                reply = ask_question(
                    st.session_state.llm,
                    st.session_state.retriever,
                    user_question
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )

    with st.chat_message("assistant"):

        st.markdown(reply)
