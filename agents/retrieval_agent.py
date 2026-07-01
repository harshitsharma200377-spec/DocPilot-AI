import os
from langchain_core.prompts import ChatPromptTemplate


class RetrievalAgent:
    """
    Retrieval Agent — handles grounded document Q&A with page-level citations.

    Uses RAG to find the most relevant document chunks and generates a
    structured, sourced answer. Supports conversational follow-ups via
    optional chat_history.
    """

    def retrieve(self, llm, retriever, query, chat_history=None):
        """
        Retrieve relevant chunks and generate a grounded answer.

        Args:
            llm:          Instantiated Groq LLM.
            retriever:    FAISS-backed LangChain retriever.
            query:        The user's question.
            chat_history: Optional list of {role, content} dicts for context.

        Returns:
            Formatted markdown string with answer, key points,
            confidence, and page-level source citations.
        """
        docs = retriever.invoke(query)

        if not docs:
            return (
                "❌ I couldn't find any relevant information in the "
                "uploaded documents."
            )

        context = "\n\n".join([doc.page_content for doc in docs])

        # Build page-level source citations
        sources = {}
        for doc in docs:
            source = os.path.basename(
                doc.metadata.get("source", "Unknown Document")
            )
            page = doc.metadata.get("page", None)
            if source not in sources:
                sources[source] = set()
            if page is not None:
                sources[source].add(page + 1)  # 0-indexed → 1-indexed

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
- If the answer is missing, say: "I couldn't find that information in the uploaded documents."
- Keep responses professional and easy to understand.
- Use recent conversation only to understand follow-up questions.

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
            question=query,
        )

        response = llm.invoke(final_prompt)

        # Format source citations with page numbers
        source_lines = []
        for src, pages in sorted(sources.items()):
            if pages:
                page_list = ", ".join(str(p) for p in sorted(pages))
                source_lines.append(f"- 📄 **{src}** — Pages: {page_list}")
            else:
                source_lines.append(f"- 📄 **{src}**")

        source_text = "\n".join(source_lines)

        return f"""
{response.content}

---

## 📚 Sources

{source_text}

**Document Chunks Used:** {len(docs)}
"""
