class ComparisonAgent:

    def compare(self, llm, retriever, doc_names=None):

        if doc_names and len(doc_names) < 2:
            return {
                "single_document": True,
                "message": "Please upload at least 2 PDF documents to perform a comparison."
            }

        docs = retriever.invoke("Compare all uploaded documents.")

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        doc_list = ""
        if doc_names:
            doc_list = "Comparing documents: " + ", ".join(doc_names) + "\n\n"

        prompt = f"""
You are DocPilot-AI.

{doc_list}Compare the uploaded documents.

Your answer should contain:

# Similarities

# Differences

# Key Findings

# Final Conclusion

Use ONLY the information below.

Context:

{context}
"""

        response = llm.invoke(prompt)

        return response.content
