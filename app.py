import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from loader import load_documents
from splitter import split_documents
from vectorstore import create_vectorstore, load_vectorstore
from rag_chain import create_rag_chain, ask_question

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Multi-PDF_RAG_Chatbot",
    page_icon="📚",
    layout="wide"
)

os.makedirs("data", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "llm" not in st.session_state:
    st.session_state.llm = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# =====================================
# Sidebar
# =====================================
with st.sidebar:
    st.title("📂 Document Upload")

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

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =====================================
# Main Page
# =====================================
st.title("📚 Multi-PDF_RAG_Chatbot")
st.write("Upload one or more PDF documents and ask questions about them.")

# =====================================
# Process Uploaded PDFs
# =====================================
if process_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("Processing documents... please wait!"):
            docs = load_documents()
            chunks = split_documents(docs)
            db = create_vectorstore(chunks)
            llm, retriever = create_rag_chain(db)
            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.rag_ready = True

        st.success(f"✅ {len(chunks)} chunks processed and ready!")

# =====================================
# Chat History
# =====================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# Chat Input
# =====================================
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if not st.session_state.rag_ready:
        reply = "⚠️ Please upload and process documents first!"
    else:
        with st.spinner("Thinking..."):
            reply = ask_question(
                st.session_state.llm,
                st.session_state.retriever,
                user_question
            )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
