from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(data_path="data/"):
    docs = []
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_path, file))
            docs.extend(loader.load())
    return docs


if __name__ == "__main__":
    documents = load_documents()
    print(f"Total pages loaded: {len(documents)}")
    print(f"Preview: {documents[0].page_content[:200]}")