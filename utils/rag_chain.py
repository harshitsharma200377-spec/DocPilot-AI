import os
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def _get_api_key() -> str:
    """Return GROQ_API_KEY from Streamlit secrets (Cloud) or env (local)."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


def create_rag_chain(vectorstore, model_name="llama-3.1-8b-instant"):
    """
    Creates the Groq LLM and document retriever.
    """

    llm = ChatGroq(
        api_key=_get_api_key(),
        model_name=model_name,
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return llm, retriever


def ask_question(llm, retriever, query, chat_history=None):
    """
    Retrieves relevant document chunks and generates an answer.
    Supports optional chat_history for conversational follow-ups.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I couldn't find any relevant information in the uploaded documents."

    context = "\n\n".join([doc.page_content for doc in docs])

    # Get unique source file names
    sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        source = os.path.basename(source)
        sources.add(source)

    # Build chat history text for context-aware answers
    history_text = ""
    if chat_history:
        history_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in chat_history[-6:]
        )

    prompt = ChatPromptTemplate.from_template(
        """
You are **DocPilot-AI**, an Intelligent AI Research Assistant.

Your responsibilities:

- Answer ONLY using the provided document context.
- Never invent facts.
- If the answer is missing, clearly say:
  "I couldn't find that information in the uploaded documents."
- Keep responses professional and easy to understand.
- Use recent conversation only to understand follow-up questions. Do not use it as a source of facts.

Your response MUST follow this format:

## Answer

(Provide the direct answer.)

## Key Points

- Point 1
- Point 2
- Point 3

## Confidence

High / Medium / Low

Context:
{context}

Recent Conversation:
{chat_history}

User Question:
{question}
"""
    )

    final_prompt = prompt.format(
        context=context,
        chat_history=history_text or "No previous conversation.",
        question=query
    )

    response = llm.invoke(final_prompt)

    source_text = "\n".join([f"- {src}" for src in sorted(sources)])

    return f"""
{response.content}

---

## Sources

{source_text}

**Document Chunks Used:** {len(docs)}
"""
