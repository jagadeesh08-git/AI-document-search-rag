📄 AI Document Search using RAG

An AI-powered Document Question Answering system built using Retrieval-Augmented Generation (RAG).
Upload documents and ask questions — answers are generated strictly from uploaded content.

This project demonstrates the practical use of:

Large Language Models (LLMs)

Vector databases

Document retrieval pipelines

Streamlit web apps

🚀 Demo

Deployed on Streamlit Cloud:
https://ai-document-search-rag-85nzddooadkfysdpwyzc6a.streamlit.app/

📸 App Preview

Add a screenshot here later:

<img width="1913" height="875" alt="image" src="https://github.com/user-attachments/assets/f90b7d31-78e0-4b6d-bab6-6efe0e34003c" />


✨ Features
📄 Multiple Document Upload

Upload PDF, TXT, and CSV files simultaneously.
All documents are automatically indexed for search.

💡 Auto Question Suggestions (Per Document)

The system generates smart question suggestions for each uploaded document by extracting keywords from document content.

Helps users:

Explore documents quickly

Understand main topics

Ask meaningful questions

🔎 Document Question Answering

Users can ask questions about uploaded files.
Answers are generated only from retrieved document chunks.

💬 Chat History Tracking

The app stores:

Previous questions

Generated answers

This allows users to review earlier interactions.

⬇️ Download Chat History

Users can download the entire session as a text file for later reference.

📚 Sidebar Document Manager

Sidebar includes:

File upload interface

Document indexing

App controls

🧠 Local LLM Inference

The system runs FLAN-T5 locally using HuggingFace Transformers.

No external API required.

🧠 How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

Upload documents

Extract text from files

Split text into chunks

Convert chunks into embeddings

Store embeddings in ChromaDB

Retrieve relevant chunks

Generate answer using FLAN-T5

🏗 Architecture
User Query
    ↓
Retriever (Chroma Vector DB)
    ↓
Relevant Document Chunks
    ↓
FLAN-T5 Language Model
    ↓
Generated Answer

🛠 Tech Stack
Frontend

Streamlit

AI / NLP

Transformers (FLAN-T5)

LangChain

Vector Database

ChromaDB

Document Processing

PyPDFLoader

TextLoader

CSVLoader

Language

Python

📦 Installation

Clone the repository:

git clone https://github.com/jagadeesh08-git/AI-document-search-rag.git
cd AI-document-search-rag


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

📂 Project Structure
ai-document-search-rag/
│
├── app.py
├── requirements.txt
├── config.toml
└── README.md

🎯 Learning Objectives

This project demonstrates:

Retrieval-Augmented Generation (RAG)

Document embeddings

Vector similarity search

Prompt-based answer generation

Streamlit deployment

LangChain integration

Local LLM usage

🔮 Future Improvements

Replace FakeEmbeddings with real embeddings

Add HuggingFace embedding models

Add OpenAI embedding option

Add multi-document citation references

Add conversation memory

Add authentication

Improve UI/UX

Deploy with GPU inference
