# DocPilot-AI

Intelligent multi-agent document research assistant built with Streamlit, LangChain, Groq, HuggingFace embeddings, and FAISS.

DocPilot-AI lets users upload PDF documents, build a local vector index, and ask natural-language questions. A planner agent routes each request to the most suitable workflow: retrieval, summarization, comparison, or quiz generation.

[Live Demo](https://docpilot-ai-cjenve4gzhwdqc5rtnra8w.streamlit.app/) | [Repository](https://github.com/harshitsharma200377-spec/DocPilot-AI) | [Author](#author)

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

## 🎯 Example Prompts

Try these prompts to see different agents in action:
* **Retrieve:** *"What is the main finding in Section 4?"*
* **Summarize:** *"Create a summary highlighting key takeaways."*
* **Compare:** *"Compare the methodologies used in document A and B."*
* **Quiz:** *"Quiz me on this document with 5 hard questions."*

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/harshitsharma200377-spec/DocPilot-AI.git
cd DocPilot-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

For Streamlit Cloud, add the same keys in the app secrets settings.

### 5. Run the app

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Notes

- The app currently supports PDF files.
- Comparison works best after uploading at least two PDFs.
- Retrieval quality depends on PDF text quality and chunk relevance.
- Scanned PDFs may require OCR before upload.

## Roadmap

- Add page-level source citations.
- Add confidence scoring for retrieved answers.
- Add OCR support for scanned PDFs.
- Add DOCX and TXT ingestion.
- Add persistent conversation storage.
- Improve evaluation for retrieval quality.

## Author

**Harshit Sharma**  
Data Analyst, Generative AI Enthusiast, and Python Developer

- [LinkedIn](https://linkedin.com/in/harshit-sharma-8b2906386)
- [GitHub](https://github.com/harshitsharma200377-spec)

## Support

If this project is useful, consider starring the [repository](https://github.com/harshitsharma200377-spec/DocPilot-AI).
