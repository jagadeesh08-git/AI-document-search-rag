import streamlit as st
import tempfile
import os
import re
import random
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================= PAGE SETUP =================
st.set_page_config(page_title="AI Document Search using RAG", layout="centered")

st.title("📄 AI Document Search using RAG")
st.write("Upload documents and ask questions based only on them.")

# ================= SESSION STATE =================
defaults = {
    "chat_history": [],
    "query": "",
    "last_files": None,
    "suggestions": [],
    "chunks": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= CLEAR CHAT =================
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.query = ""
    st.session_state.suggestions = []

    if st.session_state.chunks:
        st.session_state.suggestions = generate_suggestions(
            st.session_state.chunks
        )

    st.rerun()

# ================= SAVE FILE =================
def save_file(file):
    data = file.getvalue()
    suffix = "." + file.name.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name

# ================= LOAD DOCUMENTS =================
def load_docs(files):
    documents = []

    loader_map = {
        "pdf": PyPDFLoader,
        "txt": TextLoader,
        "csv": CSVLoader,
        "docx": Docx2txtLoader
    }

    for file in files:
        path = save_file(file)
        ext = file.name.split(".")[-1].lower()
        loader_class = loader_map.get(ext)

        if loader_class:
            docs = loader_class(path).load()
            for d in docs:
                d.metadata["source"] = file.name
            documents.extend(docs)

        os.remove(path)

    return documents

# ================= VECTOR DB =================
@st.cache_resource
def build_vector_db(files):
    docs = load_docs(files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(chunks, embeddings)
    return db, chunks

# ================= LLM =================
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

# ================= SUGGESTIONS =================
QUESTION_PATTERNS = [
    "Explain {} in detail.",
    "How is {} used in real-world applications?",
    "What are the key components of {}?",
    "Why is {} important?",
    "What challenges are related to {}?",
    "How does {} improve performance?",
    "Where can {} be applied?"
]

def generate_suggestions(chunks, max_per_doc=5):
    docs_text = {}
    suggestions = {}

    for d in chunks:
        source = d.metadata.get("source", "Unknown")
        docs_text.setdefault(source, "")
        docs_text[source] += " " + d.page_content.lower()

    for source, text in docs_text.items():
        words = re.findall(r"\b[a-zA-Z]{6,}\b", text)
        keywords = [w for w, _ in Counter(words).most_common(10)]

        random.shuffle(keywords)

        doc_questions = []
        for kw in keywords[:max_per_doc]:
            q = random.choice(QUESTION_PATTERNS).format(kw)
            doc_questions.append(q)

        suggestions[source] = doc_questions

    return suggestions

# ================= SIDEBAR =================
uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "csv", "docx"],
    accept_multiple_files=True
)

# ================= PROCESS FILES =================
if uploaded_files:
    vector_db, chunks = build_vector_db(uploaded_files)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    st.session_state.chunks = chunks

    if not st.session_state.suggestions:
        st.session_state.suggestions = generate_suggestions(chunks)

# ================= INPUT =================
st.session_state.query = st.text_input(
    "Ask a question from the document:",
    value=st.session_state.query
)

query = st.session_state.query.strip()

# ================= SHOW SUGGESTIONS =================
if uploaded_files and st.session_state.suggestions and query == "":
    st.markdown("### Suggested Questions")
    for doc, qs in st.session_state.suggestions.items():
        st.markdown(f"**📄 {doc}**")
        for q in qs:
            if st.button(q, key=doc + q):
                st.session_state.query = q
                st.rerun()

# ================= ANSWER =================
def answer_question(docs, question):
    context = " ".join(d.page_content for d in docs)

    prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    tokenizer, model = load_llm()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=200)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================= EXECUTE =================
if uploaded_files and query:
    docs = retriever.invoke(query)
    answer = answer_question(docs, query)

    st.markdown("### Answer")
    st.write(answer)

    st.session_state.chat_history.append((query, answer))

# ================= CHAT HISTORY =================
if st.session_state.chat_history:
    st.markdown("## Chat History")

    chat_text = ""
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.write(f"**Q{i}:** {q}")
        st.write(a)
        chat_text += f"Q{i}: {q}\nA{i}: {a}\n\n"

    st.download_button(
        "Download Chat History",
        chat_text,
        "chat_history.txt"
    )
