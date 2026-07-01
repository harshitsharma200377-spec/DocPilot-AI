from langchain_core.prompts import ChatPromptTemplate


class SummarizerAgent:

    def summarize(self, llm, retriever):

        docs = retriever.invoke("Summarize the uploaded documents.")

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are DocPilot-AI.

Create a professional summary of the uploaded documents.

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
