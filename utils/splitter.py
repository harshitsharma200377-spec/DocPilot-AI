from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

if __name__ == "__main__":
    from loader import load_documents
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Preview: {chunks[0].page_content[:200]}")