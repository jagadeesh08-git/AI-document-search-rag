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

# ================= HEADER =================
st.markdown("""
<style>

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #6B7280;
}

/* Sidebar general text */
section[data-testid="stSidebar"] > div {
    color: white;
}

/* Upload card */
[data-testid="stFileUploader"] {
    background-color: #F9FAFB;
    border-radius: 10px;
    padding: 12px;
    border: 2px solid #4F46E5;
}

/* Fix uploader text visibility */
[data-testid="stFileUploader"] * {
    color: #111827 !important;
}

/* File name */
[data-testid="stFileUploaderFile"] {
    color: #111827 !important;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)



# ================= RECTANGLE HEADER =================
st.markdown("""
<div style="
    background-color:#4F46E5;
    padding:18px;
    border-radius:10px;
    margin-bottom:20px;
">
    <h1 style="color:white; margin:0;">
        📄 AI Document Search using RAG
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Upload one or more documents and ask questions.  
Answers are generated **strictly from uploaded documents**.
""")

# ================= SESSION STATE =================
defaults = {
    "chat_history": [],
    "query": "",
    "last_files": None,
    "suggestions": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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

    for file in files:
        path = save_file(file)
        ext = file.name.split(".")[-1].lower()

        loader_map = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "csv": CSVLoader,
            "docx": Docx2txtLoader
        }

        loader_class = loader_map.get(ext)

        if loader_class:
            try:
                docs = loader_class(path).load()
                for d in docs:
                    d.metadata["source"] = file.name
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")

        os.remove(path)

    return documents

# ================= VECTOR DB =================
@st.cache_resource(show_spinner=False)
def build_vector_db(files):
    docs = load_docs(files)
    if not docs:
        raise ValueError("No readable text found in uploaded documents.")

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
@st.cache_resource(show_spinner=False)
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
    "Where can {} be applied?",
    "What role does {} play in the document?",
    "How is {} implemented?",
    "What does the document mention about {}?"
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
        random.shuffle(QUESTION_PATTERNS)

        doc_questions = []
        used = set()

        for kw in keywords:
            pattern = random.choice(QUESTION_PATTERNS)
            q = pattern.format(kw)

            if q not in used:
                doc_questions.append(q)
                used.add(q)

            if len(doc_questions) >= max_per_doc:
                break

        suggestions[source] = doc_questions

    return suggestions

# ================= SIDEBAR =================
st.sidebar.title("AI Document Search")

uploaded_files = st.sidebar.file_uploader(
    "Upload document(s)",
    type=["pdf", "txt", "csv", "docx"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.caption("RAG Document Assistant")

# ================= FILE CHANGE RESET =================
current_files = tuple(f.name for f in uploaded_files) if uploaded_files else None
if current_files != st.session_state.last_files:
    st.session_state.chat_history.clear()
    st.session_state.query = ""
    st.session_state.suggestions.clear()
    st.cache_resource.clear()
    st.session_state.last_files = current_files

# ================= PROCESS FILES =================
if uploaded_files:
    with st.spinner("Indexing documents..."):
        vector_db, chunks = build_vector_db(uploaded_files)
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    if not st.session_state.suggestions:
        st.session_state.suggestions = generate_suggestions(chunks)

# ================= INPUT =================
st.session_state.query = st.text_input(
    "Ask a question from the document:",
    value=st.session_state.query,
    placeholder="Type a complete question"
)

col1, col2 = st.columns(2)
with col1:
    ask_clicked = st.button("🔎 Ask")
with col2:
    clear_q = st.button("🧹 Clear Question")

if clear_q:
    st.session_state.query = ""
    st.rerun()

query = st.session_state.query.strip()

# ================= SUGGESTIONS =================
if uploaded_files and st.session_state.suggestions and query == "":
    st.markdown("💡 Suggested Questions")
    for doc, qs in st.session_state.suggestions.items():
        st.markdown(f"### 📄 {doc}")
        for q in qs:
            if st.button(q, key=doc + q):
                st.session_state.query = q
                st.rerun()

# ================= ANSWER =================
def answer_question(docs, question):
    context = " ".join(d.page_content for d in docs)

    if len(context.strip()) < 30:
        return "Information not found in the uploaded documents."

    prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    tokenizer, model = load_llm()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=200)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ================= EXECUTE =================
if uploaded_files and ask_clicked and len(query.split()) >= 3:
    docs = retriever.invoke(query)

    if docs:
        answer = answer_question(docs, query)
        st.markdown("### 🧠 Answer")
        st.write(answer)

        if "not found" not in answer.lower():
            st.session_state.chat_history.append((query, answer))

# ================= CHAT HISTORY =================
if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}: {q}**")
        st.write(a)

    chat_text = ""
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        chat_text += f"Q{i}: {q}\nA{i}: {a}\n\n"

    st.download_button(
        label="⬇️ Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )
