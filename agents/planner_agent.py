import os
import streamlit as st
from langchain_groq import ChatGroq


class PlannerAgent:

    def __init__(self):
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

    def plan(self, query, chat_history=None):

        history_text = ""
        if chat_history:
            for msg in chat_history[-4:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"

        history_section = f"\nRecent conversation:\n{history_text}\n" if history_text else ""

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

Return ONLY one word: retrieve / summarize / compare / quiz
{history_section}
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
