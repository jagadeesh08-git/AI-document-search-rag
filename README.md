AI Document Search using RAG 📄🤖

An interactive Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions.
The system retrieves relevant content from uploaded files and generates answers using a language model.

Built with Streamlit + LangChain + ChromaDB + Transformers.

Features

Upload multiple documents (PDF, TXT, CSV)

Document chunking and indexing

Vector database retrieval

AI-generated answers from document context

Suggested questions per document

Chat history tracking

Download chat history

Clean UI with sidebar controls

Tech Stack

Streamlit

LangChain

ChromaDB

HuggingFace Transformers

PyPDF

Pandas

NumPy

Torch

Project Structure
AI-DOCU-APP/
│
├── app.py
├── requirements.txt
└── .streamlit/
    └── config.toml

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/AI-document-search-rag.git
cd AI-document-search-rag


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

How it Works

User uploads documents

Documents are split into chunks

Chunks are stored in a vector database

User asks a question

Relevant chunks are retrieved

LLM generates answer from retrieved context

This is the Retrieval-Augmented Generation (RAG) pipeline.

Example Use Cases

Resume/document search assistant

Research document Q&A

Study material search

Knowledge-base assistant

Report analysis tool
