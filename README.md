# 📚 LLM-Powered Study Assistant (RAG)

An AI-powered study assistant that allows users to upload one or more PDF documents and ask questions in natural language. The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant content from uploaded documents and generate accurate responses using Llama 3 via Groq.

## 🚀 Features

- 📄 Upload and analyze multiple PDF documents
- 🔍 Semantic search using FAISS vector database
- 🤖 Context-aware question answering with Llama 3
- 🧠 Conversation memory for follow-up questions
- 💬 ChatGPT-style interactive interface
- 📚 Multi-document retrieval
- 📥 Download chat history
- ⚡ Fast retrieval using Sentence Transformers embeddings

## 🛠️ Tech Stack

- Python
- Streamlit
- Groq API (Llama 3)
- FAISS
- Sentence Transformers
- LangChain
- PyPDF
- python-dotenv

## 📂 Project Architecture

```text
PDF Upload
    ↓
Text Extraction (PyPDF)
    ↓
Chunking
    ↓
Embeddings (Sentence Transformers)
    ↓
FAISS Vector Store
    ↓
Semantic Retrieval
    ↓
Groq Llama 3
    ↓
Answer Generation
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/dharinir-dev/study-assistant.git
cd study-assistant
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create .env File

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

### Run Application

```bash
streamlit run app.py
```

## 📊 Key Features Implemented

- PDF text extraction
- Recursive text chunking
- Embedding generation
- FAISS vector indexing
- Semantic similarity search
- RAG-based response generation
- Conversation memory
- Multi-PDF support
- Modern Streamlit UI

## 🎯 Future Improvements

- Page-level source citations
- PDF summarization
- OCR support for scanned PDFs
- User authentication
- Cloud database integration
- Export answers as PDF


## 👩‍💻 Author

**Dharini**

GitHub: https://github.com/dharinir-dev

