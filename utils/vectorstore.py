from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings


def create_vectorstore(chunks):
    """
    Creates a FAISS vectorstore from chunked document text.
    Guards against empty chunk lists to prevent IndexError.
    """
    if not chunks:
        raise ValueError(
            "Cannot create vectorstore: chunk list is empty. "
            "Please ensure uploaded documents contain readable text."
        )

    db = FAISS.from_documents(chunks, get_embeddings())
    db.save_local("faiss_db")
    return db


def load_vectorstore():
    """Loads an existing FAISS vectorstore from local disk."""
    return FAISS.load_local(
        "faiss_db",
        get_embeddings(),
        allow_dangerous_deserialization=True
    )
