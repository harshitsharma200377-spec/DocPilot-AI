<div align="center">

# 🧭 DocPilot-AI

### Intelligent Multi-Agent Document Research Assistant

**Upload documents. Ask naturally. Let specialized AI agents plan, retrieve, summarize, and compare — with grounded, citation-backed answers.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=flat-square)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=flat-square)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Built for the Kaggle Competition:** *AI Agents — Intensive Vibe Coding Capstone Project with Google*

[🌐 Live Demo](https://docpilot-ai-cjenve4gzhwdqc5rtnra8w.streamlit.app/) · [📂 Repository](https://github.com/harshitsharma200377-spec/DocPilot-AI) · [👤 Author](#-author)

</div>

---

## 🧠 What Makes This an *Agent* System (Not Just RAG)

Most document Q&A tools run a single fixed pipeline: embed → retrieve → answer. DocPilot-AI instead uses a **Planner Agent** that reads the user's intent and dynamically routes the request to the specialized agent best suited to handle it — retrieval, summarization, or comparison — each with its own prompting strategy and output structure.

This means the system *reasons about what kind of task it's facing* before acting, rather than treating every question the same way.

---

## 🤖 Agent Architecture

```
                         ┌─────────────────────┐
                         │      User Query       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Planner Agent      │
                         │ (Groq LLM — intent     │
                         │  classification)       │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │  Retrieval Agent   │ │ Summarizer Agent  │ │ Comparison Agent  │
    │  FAISS semantic     │ │ Structured multi-  │ │ Cross-document     │
    │  search + grounded  │ │ section summaries  │ │ analysis tables    │
    │  citations          │ │                    │ │                    │
    └──────────┬─────────┘ └──────────┬─────────┘ └──────────┬─────────┘
               │                       │                       │
               └───────────────────────┼───────────────────────┘
                                        ▼
                         ┌─────────────────────┐
                         │  Context-Aware Answer │
                         │   (Streamlit UI)       │
                         └─────────────────────┘
```

### Agent Roles

| Agent | Responsibility |
|---|---|
| **Planner Agent** | Uses Groq LLM to classify the user's intent into `retrieve`, `summarize`, `compare`, or `quiz`, and routes to the correct downstream agent. |
| **Retrieval Agent** | Runs semantic search over the FAISS vector store, pulls the most relevant chunks, and generates grounded answers using Groq inference. |
| **Summarizer Agent** | Produces structured summaries (Executive Summary, Key Findings, Insights, Action Items, Conclusion) instead of a single paragraph dump. |
| **Comparison Agent** | Compares two or more uploaded documents, highlighting similarities, differences, and generating comparative tables. |

---

## ✨ Features

- 📄 Upload and process multiple PDF documents simultaneously
- 🧩 Automatic text extraction and intelligent chunking
- 🔎 Semantic search powered by HuggingFace embeddings + FAISS
- 🧭 LLM-driven Planner Agent for dynamic task routing
- 💬 Multi-document, context-aware question answering
- 📝 Structured summarization (not just raw text compression)
- 📊 Multi-document comparison with tabular output
- ⚡ Low-latency inference via Groq LLaMA 3.1
- 🖥️ Clean, professional Streamlit interface

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python** | Core language |
| **Streamlit** | Web application / UI layer |
| **LangChain** | Agent orchestration and RAG pipeline |
| **Groq (LLaMA 3.1)** | LLM inference for planning, retrieval, summarization, comparison |
| **HuggingFace** | Embedding models |
| **FAISS** | Vector similarity search |
| **PyPDF** | PDF text extraction |
| **python-dotenv** | Environment configuration |

---

## 🗂️ Project Structure

```
DocPilot-AI/
│
├── agents/
│   ├── planner_agent.py        # Intent classification & routing
│   ├── summarizer_agent.py     # Structured summarization
│   └── comparison_agent.py     # Multi-document comparison
│
├── utils/
│   ├── loader.py                # PDF text extraction
│   ├── splitter.py              # Document chunking
│   ├── embeddings.py            # HuggingFace embedding generation
│   ├── vectorstore.py           # FAISS vector store management
│   ├── rag_chain.py             # Retrieval + grounded generation
│   └── memory.py                # Conversation memory
│
├── app.py                       # Streamlit UI entry point
├── requirements.txt
├── README.md
└── .env                         # (not committed) API keys
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/harshitsharma200377-spec/DocPilot-AI.git
cd DocPilot-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux / Mac**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

### 5. Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 💡 Example Workflow

1. Upload one or more PDF documents.
2. Click **Process Documents** to build the FAISS index.
3. Ask a question in natural language — the Planner Agent decides how to handle it.

**Try prompts like:**

```
Summarize this document.
What are the key findings in Chapter 3?
Compare the conclusions of both PDFs.
Who is the author, and what is their main argument?
Give me the action items from this report.
```

---

## 🎯 Use Cases

- 📚 Research paper analysis
- 📑 Legal document review
- 📊 Business & company report analysis
- 📘 Study notes and exam prep
- 📃 Technical documentation Q&A
- 📖 Long-form book/report comprehension

---

## 🔮 Roadmap

- [ ] Source citations with page-level references
- [ ] Confidence scoring on retrieved answers
- [ ] Quiz Agent (auto-generated Q&A from documents)
- [ ] Persistent conversational memory across sessions
- [ ] OCR support for scanned PDFs
- [ ] DOCX and TXT ingestion
- [ ] Multi-modal document understanding (tables, images)
- [ ] User authentication for multi-user deployments

---

## 👨‍💻 Author

**Harshit Sharma**
Data Analyst · Generative AI Enthusiast · Python Developer

- 🔗 [LinkedIn](https://linkedin.com/in/harshit-sharma-8b2906386)
- 💻 [GitHub](https://github.com/harshitsharma200377-spec)

---

## ⭐ Support

If this project was useful or interesting, consider leaving a ⭐ on the [repository](https://github.com/harshitsharma200377-spec/DocPilot-AI) — it helps others discover it.
