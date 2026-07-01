# 🧭 DocPilot-AI
> **Intelligent Multi-Agent Document Intelligence System**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badge-gradient.svg)](https://docpilot-ai-cjenve4gzhwdqc5rtnra8w.streamlit.app/)
[![Groq Powered](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)
[![FAISS Store](https://img.shields.io/badge/VectorStore-FAISS-green.svg)](https://github.com/facebookresearch/faiss)

DocPilot-AI is a production-inspired document intelligence application built as a capstone project for the Kaggle & Google course/competition. Unlike static Retrieval-Augmented Generation (RAG) pipelines, DocPilot-AI implements a true **Multi-Agent Orchestration Architecture** where user queries are dynamically analyzed and routed to specialized specialized agents.

---

## 🧠 System Architecture

Instead of routing every query through a single pipeline, DocPilot-AI uses a **Planner Agent** to inspect incoming user requests, classify the user's intent, and delegate execution to the most suited downstream agent.

```text
               [User Query]
                    │
                    ▼
           ┌─────────────────┐
           │  Planner Agent  │  <─── (Routes dynamically based on intent)
           └────────┬────────┘
                    │
      ┌─────────────┼─────────────┬─────────────┐
      │             │             │             │
      ▼             ▼             ▼             ▼
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│ Retrieval │ │Summarizer │ │Comparison │ │   Quiz    │
│   Agent   │ │   Agent   │ │   Agent   │ │   Agent   │
└─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
      │             │             │             │
      └─────────────┼─────────────┴─────────────┘
                    │
                    ▼
            [Streamlit UI]
```

### The Agent Registry
* **Planner Agent:** Inspects the query and decides the downstream intent using few-shot classifications (`llama-3.1-8b-instant`).
* **Retrieval Agent:** Handles grounded QA over document chunks. Formats responses with semantic highlights, confidence levels, and **page-level sources**.
* **Summarizer Agent:** Generates executive-style structured document summaries (Executive Summary, Key Findings, Insights, Conclusion).
* **Comparison Agent:** Performs cross-document comparison tables and analyzes differences/similarities when multiple files are uploaded.
* **Quiz Agent:** Generates interactive study quizzes containing MCQs, True/False, and short answers derived from the corpus.

---

## 🛠️ Tech Stack & Design Choices

* **Streamlit UI:** Formatted with custom CSS themes to deliver a modern, premium dark-mode dashboard.
* **LangChain:** Used to coordinate agent calls, wrap prompt templates, and construct conversational memory.
* **Groq Inference:** Powering all agents with ultra-fast inference using `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, or `mixtral-8x7b-32768`.
* **HuggingFace Embeddings:** Utilizing `all-MiniLM-L6-v2` for local, high-quality vector embeddings.
* **FAISS Vector Index:** In-memory vector store for fast, lightweight local semantic searches.

---

## ✨ Advanced Features

* **LLM Model Switcher:** Instantly swap between models directly from the UI sidebar without rebuilding the vector store.
* **Page-Level Citations:** Source citations list exact page numbers (e.g. `Page: 3, 7`) extracted from PDF metadata.
* **Chat History Memory:** Lightweight sliding-window memory maintains conversation context for multi-turn Q&A.
* **Download Chat History:** Export complete conversation logs as clean Markdown documents with a single click.

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/harshitsharma200377-spec/DocPilot-AI.git
cd DocPilot-AI
```

### 2. Set up virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a local `.env` file:
```env
GROQ_API_KEY=your-api-key-here
```
*Note: For Streamlit Cloud deployments, add `GROQ_API_KEY` to the app's Secrets manager.*

### 5. Launch the application
```bash
streamlit run app.py
```

---

## 🎯 Example Prompts

Try these prompts to see different agents in action:
* **Retrieve:** *"What is the main finding in Section 4?"*
* **Summarize:** *"Create a summary highlighting key takeaways."*
* **Compare:** *"Compare the methodologies used in document A and B."*
* **Quiz:** *"Quiz me on this document with 5 hard questions."*

---

## 👤 Author

**Harshit Sharma**  
*Data Analyst & Generative AI Engineer*
* [LinkedIn](https://linkedin.com/in/harshit-sharma-8b2906386)
* [GitHub](https://github.com/harshitsharma200377-spec)
