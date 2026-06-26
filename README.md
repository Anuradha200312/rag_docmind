# 🧠 DocMind AI — AI-Powered Hybrid Document Assistant

Welcome to **DocMind AI**, a functional MVP designed for the **Techverse AI-powered Document Assistant** assignment. This application features a **Hybrid Ingestion Pipeline** and **PostgreSQL Database Integration**, packaged within a premium, responsive **Streamlit** user interface.

It dynamically routes documents depending on their token size (utilizing local ChromaDB for small documents and Qdrant Cloud for larger documents), maintains persistent chat history across sessions, and ensures accurate document-grounded QA.

---

## ✨ Features & Requirements Met

1. **📄 PDF Ingestion & Processing**: Extracts page-by-page text from PDF uploads and processes them using custom chunks (chunk size: 4000 characters, overlap: 800 characters).
2. **🔀 Hybrid Pipeline Routing**:
   - **≤ 20,000 tokens**: Automatically routes to the **ChromaDB Pipeline** (Local in-memory store for sub-second query latency).
   - **> 20,000 tokens**: Automatically routes to the **Qdrant RAG Pipeline** (uses `SentenceTransformer` local embeddings uploaded to Qdrant Cloud).
3. **❓ Grounded Document QA**: Queries are resolved using context retrieved from the vector store, passed to a Groq LLM (`llama3-8b-8192`). 
4. **🚫 Out-of-Document Handling**: If the answer to a question cannot be resolved using the context from the document, the assistant gracefully handles it and notifies the user rather than hallucinating.
5. **💬 Persistent Chat History**: Stores and manages multi-turn chats, chat sessions, and history logs per user in a PostgreSQL database.
6. **🎨 Premium UI**: Features custom-styled glassmorphism cards, animated status widgets, and linear-gradient styling within a Streamlit dashboard.

---

## 📂 Project Architecture

```
doc_mind/
├── .env                # API keys, database URLs, and configuration settings
├── README.md           # Setup, run instructions, and submission checklist
├── app.py              # Streamlit entry point (UI and session logic)
├── database.py         # PostgreSQL database schema and synchronous CRUD helpers
├── pdf_helper.py       # PDF text extraction, token counting, and text chunking
├── vector_store.py     # Embedding generation, ChromaDB, and Qdrant Cloud operations
└── workflow.py         # LangGraph workflow definitions (Ingestion & Query graphs)
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.10+
- PostgreSQL database instance running locally or on a cloud service (e.g., Supabase, Neon, or RDS)

### 2. Installation
Clone the repository, navigate to the folder, activate your virtual environment, and install the dependencies:

```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all dependencies
pip install streamlit langgraph langchain-groq langchain-core langchain-text-splitters tiktoken pypdf sqlalchemy psycopg2-binary chromadb qdrant-client sentence-transformers python-dotenv
```

### 3. Environment Variables Configuration
Create or configure the `.env` file in the root folder with the following variables:

```ini
# Groq LLM API Key
GROQ_API_KEY="your_groq_api_key"

# PostgreSQL Database Connection URL (e.g., Supabase, Neon, or Local)
DATABASE_URL="postgresql://postgres:password@localhost:5432/docmind"

# Qdrant Vector Database Settings for RAG
QDRANT_URL="your_qdrant_cloud_url"
QDRANT_API_KEY="your_qdrant_api_key"

# Defaults & Pipeline Configurations
TOKEN_THRESHOLD=20000
EMBEDDING_MODEL="all-MiniLM-L6-v2"
EMBEDDING_DIMENSION=384
MAX_CHUNK_SIZE=800
CHUNK_OVERLAP=150
RAG_TOP_K=5
```

---

## 🚀 How to Run the App

1. Ensure **PostgreSQL** is running and the database matches the connection URL in `.env`.
2. Start the Streamlit server from your terminal:
   ```powershell
   streamlit run app.py
   ```
3. Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment (Streamlit Community Cloud)

To deploy to **Streamlit Community Cloud**:
1. Push this repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch, and set the Main file path to `app.py`.
4. Under **Advanced Settings**, paste the contents of your `.env` file into the **Secrets** section:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   DATABASE_URL = "postgresql://postgres:password@localhost:5432/docmind"
   QDRANT_URL = "your_qdrant_cloud_url"
   QDRANT_API_KEY = "your_qdrant_api_key"
   TOKEN_THRESHOLD = 20000
   EMBEDDING_MODEL = "all-MiniLM-L6-v2"
   EMBEDDING_DIMENSION = 384
   MAX_CHUNK_SIZE = 800
   CHUNK_OVERLAP = 150
   ```
5. Click **Deploy**.

---

## 📝 Submission Deliverables

- **GitHub Repository**: `[Insert Repository URL]`
- **Published Streamlit Link**: `[Insert Streamlit Community Cloud URL]`
- **Demo Video (3–5 minutes)**: `[Insert Demo Video Link]`
