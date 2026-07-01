from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def create_rag_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return llm, retriever

def ask_question(llm, retriever, query):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    from vectorstore import load_vectorstore
    db = load_vectorstore()
    llm, retriever = create_rag_chain(db)
    answer = ask_question(llm, retriever, "What is calisthenics?")
    print(f"✅ Answer: {answer}")