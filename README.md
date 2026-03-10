# 📄 Chat with PDF — RAG Based AI Document Assistant

An AI-powered web application that allows users to upload a PDF and ask questions about its content. Built using **Retrieval-Augmented Generation (RAG)** with Google Gemini and FAISS.

---

## 🚀 Features

- 📄 Upload and process any PDF document
- 💬 Ask natural language questions about the document
- 🔍 Semantic search using vector embeddings
- 🧠 Context-aware answers via Google Gemini
- 💾 Efficient document indexing using FAISS
- 🗂 Conversation history for follow-up questions
- ⚡ Clean chat interface built with Streamlit

---

## 🧠 How It Works (RAG Pipeline)

**Phase 1 — Indexing (when PDF is uploaded):**
```
PDF → PyPDF2 → Text Chunks → Gemini Embeddings → FAISS Vector Store
```

**Phase 2 — Retrieval (when user asks a question):**
```
Question → Embedding → FAISS Similarity Search → Top 3 Chunks → Gemini LLM → Answer
```

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM | Google Gemini (gemini-2.5-flash) |
| Embeddings | Gemini Embedding Model |
| Vector Database | FAISS |
| Framework | Streamlit |
| PDF Processing | PyPDF2 |
| Text Splitting | LangChain Text Splitters |

---

## 📂 Project Structure
```
pdf-rag-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment variables
├── .gitignore
└── faiss_index/           # Generated vector database (auto-created)
```

---

## ⚙️ Setup

**1. Clone the repository**
```bash
git clone https://github.com/priyanshu812/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Gemini API Key**

Create a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```
Get your free API key at: https://aistudio.google.com

**5. Run the app**
```bash
streamlit run app.py
```
App runs at: http://localhost:8501

---

## 📚 Key Concepts

- Retrieval-Augmented Generation (RAG)
- Vector embeddings and similarity search
- Context-aware LLM prompting
- Conversational memory management
- FAISS vector database

---

## 🔮 Future Improvements

- Support for multiple PDFs
- Source citations for answers
- Streaming responses
- Cloud deployment

---

## 👨‍💻 Author

**Priyanshu Soni** — B.Tech CSE (AI/ML)

⭐ If you found this helpful, give the repo a star!