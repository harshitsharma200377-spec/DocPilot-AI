from langchain_community.vectorstores import FAISS
from embeddings import get_embeddings

def create_vectorstore(chunks):
    db = FAISS.from_documents(chunks, get_embeddings())
    db.save_local("faiss_db")
    return db

def load_vectorstore():
    return FAISS.load_local(
        "faiss_db",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )