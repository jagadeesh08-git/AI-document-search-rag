# 📄 AI Document Search using RAG

An AI-powered **Document Question Answering system** built using **Retrieval-Augmented Generation (RAG)**.
Upload documents and ask questions — answers are generated **strictly from the uploaded content**.

This project demonstrates practical use of **LLMs + vector databases + document retrieval** in a simple web application.

---

## 🚀 Demo

Deployed with Streamlit Cloud
👉 Add your app link here

Example:

```
[https://your-app-name.streamlit.app](https://ai-document-search-rag-85nzddooadkfysdpwyzc6a.streamlit.app/)
```

---
## 📸 App Preview

<img width="1917" height="870" alt="image" src="https://github.com/user-attachments/assets/d841b7fa-8214-486d-8387-f8e5942b076d" />



## ✨ Features

* Upload **PDF, TXT, and CSV documents**
* Ask questions about uploaded files
* Retrieval-Augmented Generation (RAG) pipeline
* Automatic **document indexing**
* **Suggested questions generator**
* Chat history tracking
* Download chat history
* Clean Streamlit UI
* Local LLM inference using FLAN-T5

---

## 🧠 How It Works

The application follows the RAG pipeline:

1. Upload documents
2. Extract text using document loaders
3. Split text into chunks
4. Convert chunks into embeddings
5. Store embeddings in Chroma vector database
6. Retrieve relevant chunks for the question
7. Generate answer using FLAN-T5 model

---

## 🏗 Architecture

```
User Query
    ↓
Retriever (Chroma Vector DB)
    ↓
Relevant Document Chunks
    ↓
FLAN-T5 Language Model
    ↓
Generated Answer
```

---

## 🛠 Tech Stack

**Frontend**

* Streamlit

**AI / NLP**

* Transformers (FLAN-T5)
* LangChain

**Vector Database**

* ChromaDB

**Document Processing**

* PyPDF
* TextLoader
* CSVLoader

**Language**

* Python

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-document-search-rag.git
cd ai-document-search-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
ai-document-search-rag/
│
├── app.py
├── requirements.txt
├── config.toml
└── README.md
```

---

## 🎯 Learning Objectives

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Document embeddings
* Vector similarity search
* Prompt-based generation
* Streamlit app deployment
* LLM integration with LangChain

---

## 🔮 Future Improvements

* Replace FakeEmbeddings with real embeddings
* Add OpenAI / HuggingFace embedding models
* Add multi-document citation references
* Add conversation memory
* Add authentication
* Improve UI/UX

---
