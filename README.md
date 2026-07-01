# DocPilot-AI

Intelligent multi-agent document research assistant built with Streamlit, LangChain, Groq, HuggingFace embeddings, and FAISS.

DocPilot-AI lets users upload PDF documents, build a local vector index, and ask natural-language questions. A planner agent routes each request to the most suitable workflow: retrieval, summarization, comparison, or quiz generation.

[Live Demo](https://docpilot-ai-cjenve4gzhwdqc5rtnra8w.streamlit.app/) | [Repository](https://github.com/harshitsharma200377-spec/DocPilot-AI) | [Author](#author)

## Why This Is an Agent System

Most document Q&A apps use a fixed pipeline:

```text
embed -> retrieve -> answer
```

DocPilot-AI adds a planner agent before the document workflow. The planner reads the user's request and chooses the correct downstream agent:

```text
User query
  |
  v
Planner Agent
  |
  +--> Retrieval Agent   -> grounded document Q&A
  +--> Summarizer Agent  -> structured summaries
  +--> Comparison Agent  -> multi-document comparison
  +--> Quiz Agent        -> generated study questions
  |
  v
Streamlit response
```

## Features

- Upload and process multiple PDF files.
- Extract text and split documents into searchable chunks.
- Build a FAISS vector store using HuggingFace embeddings.
- Route user requests with a Groq-powered planner agent.
- Answer document questions with grounded context.
- Summarize uploaded documents in a structured format.
- Compare two or more PDFs.
- Generate quizzes with multiple-choice, true/false, and short-answer questions.
- Keep lightweight recent chat history for follow-up questions.
- Use a polished dark Streamlit interface.

## Tech Stack

| Technology | Role |
| --- | --- |
| Python | Core language |
| Streamlit | Web app and UI |
| LangChain | Retrieval and orchestration helpers |
| Groq | LLM inference |
| HuggingFace | Embedding model |
| FAISS | Vector similarity search |
| PyPDF | PDF extraction |
| python-dotenv | Local environment variables |

## Project Structure

```text
DocPilot-AI/
├── agents/
│   ├── __init__.py
│   ├── planner_agent.py
│   ├── summarizer_agent.py
│   ├── comparison_agent.py
│   └── quiz_agent.py
├── utils/
│   ├── __init__.py
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── rag_chain.py
│   └── memory.py
├── screenshots/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Runtime folders such as `data/` and `faiss_db/` are created by the app and should not be committed.

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

## Example Prompts

```text
Summarize this document.
What are the key findings in chapter 3?
Compare the conclusions of both PDFs.
Who is the author, and what is the main argument?
Create a medium difficulty quiz from this document.
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
