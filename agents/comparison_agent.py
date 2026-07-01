class ComparisonAgent:

    def compare(self, llm, retriever):

        docs = retriever.invoke("Compare all uploaded documents.")

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are DocPilot-AI.

Compare the uploaded documents.

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
