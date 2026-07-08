# 📄 PDF RAG Chatbot with Langfuse Evaluation

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based **only on the uploaded PDF document**. The project integrates **Groq LLM**, **ChromaDB**, **Sentence Transformers**, **Streamlit**, and **Langfuse** for tracing and automated LLM evaluation.

---

## 🚀 Features

- 📄 Upload any PDF document
- 🔍 Semantic search using Sentence Transformer embeddings
- 🧩 Automatic text chunking
- 🗂️ ChromaDB vector database for retrieval
- 🤖 Groq LLM integration for response generation
- 💬 Interactive Streamlit chatbot
- 📊 Langfuse tracing for every query
- 📈 Automated evaluation using Langfuse Managed Evaluators
- 🔒 Answers are restricted to the uploaded document only
- ⚡ Fast retrieval and response generation

---

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │   PDF Upload     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   PDF Loader     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Text Chunking    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Embedding Model  │
                    │ all-MiniLM-L6-v2 │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    ChromaDB      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Retriever      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Prompt Builder   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Groq LLM       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Streamlit UI     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Langfuse      │
                    │ Tracing & Eval   │
                    └──────────────────┘
```

---

# 🛠️ Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| UI | Streamlit |
| LLM | Groq (GPT-OSS-20B / Llama Models) |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Prompt Framework | LangChain |
| Observability | Langfuse |
| Evaluation | Langfuse Managed Evaluators |

---

# 📂 Project Structure

```text
rag_chatbot/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── data/
│
├── rag/
│   ├── pdf_loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vectordb.py
│   ├── retriever.py
│   ├── prompt.py
│   ├── llm.py
│   └── pipeline.py
│
├── tracing/
│   ├── __init__.py
│   └── langfuse_callback.py
│
└── utils/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git

cd <repository-name>
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key

LANGFUSE_PUBLIC_KEY=your_langfuse_public_key

LANGFUSE_SECRET_KEY=your_langfuse_secret_key

LANGFUSE_HOST=https://cloud.langfuse.com
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application opens in your browser.

1. Upload a PDF.
2. Ask questions.
3. Receive answers grounded only in the uploaded document.

---

# 📊 Langfuse Tracing

Every user query is automatically traced.

Each trace contains:

- User Question
- Retrieved Context
- Prompt
- LLM Response
- Token Usage
- Latency
- Metadata

---

# 📈 Managed Evaluators

The project supports Langfuse Managed Evaluators including:

- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall
- Context Relevance
- Hallucination
- Helpfulness
- Conciseness
- Correctness
- Toxicity

These evaluations are visible in the Langfuse dashboard and are not displayed in the Streamlit interface.

---

# 🔮 Future Improvements

- Multi-PDF retrieval
- Hybrid search (BM25 + Vector Search)
- Cross-encoder reranking
- Streaming responses
- Persistent vector storage
- Conversation memory
- Docker deployment
- User authentication
