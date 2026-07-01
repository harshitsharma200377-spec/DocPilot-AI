from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


def create_rag_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return llm, retriever


def ask_question(llm, retriever, query):

    docs = retriever.invoke(query)

    if not docs:
        return "I couldn't find relevant information in the uploaded documents."

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are DocPilot-AI.

You are an intelligent AI Research Agent.

Your responsibilities:

- Answer ONLY using the provided document context.
- If the answer isn't present, clearly say:
"I couldn't find that information in the uploaded documents."

- Never make up information.

- Be concise.

- Use bullet points whenever appropriate.

- End every answer with:

📚 Sources:
Mention how many document chunks were used.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content + f"\n\n📚 Sources Used: {len(docs)} document chunks"
