from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class PlannerAgent:

    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

    def plan(self, query):

        prompt = f"""
You are the planning brain of DocPilot-AI.

Your job is to choose the SINGLE best agent.

Available Agents:

1. retrieve
- Answer questions from uploaded PDFs.

2. summarize
- Summarize one or more uploaded documents.

3. compare
- Compare multiple uploaded documents.

4. quiz
- Generate quiz questions from uploaded documents.

Return ONLY one word.

retrieve
summarize
compare
quiz

User Request:

{query}
"""

        decision = self.llm.invoke(prompt).content.lower().strip()

        if "summarize" in decision:
            return "summarize"

        elif "compare" in decision:
            return "compare"

        elif "quiz" in decision:
            return "quiz"

        return "retrieve"
