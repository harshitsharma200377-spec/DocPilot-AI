from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


if __name__ == "__main__":
    embeddings = get_embeddings()
    test = embeddings.embed_query("hello world")
    print("Embeddings working!")
    print(f"Vector size: {len(test)}")
