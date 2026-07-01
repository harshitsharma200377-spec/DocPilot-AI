class QuizAgent:

    def generate_quiz(self, llm, retriever, difficulty="Medium", num_questions=5):

        docs = retriever.invoke("Generate quiz questions from the uploaded documents.")

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are DocPilot-AI Quiz Generator.

Generate a quiz based on the document content below.

Difficulty: {difficulty}
Number of questions: {num_questions}

Create a MIX of:
- Multiple Choice Questions (MCQ)
- True/False questions
- Short Answer questions

Format your response EXACTLY like this:

# Multiple Choice Questions

1. Question text?
   A) Option A
   B) Option B
   C) Option C
   D) Option D
   Answer: A

# True / False

1. Statement here.
   Answer: True

# Short Answer

1. Question here?
   Answer: Suggested answer here.

Context:

{context}
"""

        response = llm.invoke(prompt)

        return response.content
