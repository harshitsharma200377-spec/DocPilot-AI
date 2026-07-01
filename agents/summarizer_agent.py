class SummarizerAgent:

    def summarize(self, llm, retriever, doc_names=None):

        docs = retriever.invoke("Summarize the uploaded documents.")

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        doc_list = ""
        if doc_names:
            doc_list = "Documents being summarized: " + ", ".join(doc_names) + "\n\n"

        prompt = f"""
You are DocPilot-AI.

{doc_list}Create a professional summary of the uploaded documents.

Your response MUST contain:

# Executive Summary

# Key Topics

# Important Insights

# Conclusion

Context:

{context}
"""

        response = llm.invoke(prompt)

        return response.content
