import os
import tempfile
import numpy as np
import faiss
import streamlit as st
from google import genai
from fastembed import TextEmbedding
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚", layout="wide")

# ---------- API key ----------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = st.sidebar.text_input("Gemini API key", type="password")

@st.cache_resource
def load_model():
    return TextEmbedding("BAAI/bge-small-en-v1.5")

model = load_model()

# ---------- Sidebar ----------
st.sidebar.title("📚 PDF Q&A")
uploaded = st.sidebar.file_uploader("Upload a PDF", type="pdf")
k = st.sidebar.slider("Chunks to retrieve", 2, 8, 4)
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []

# ---------- Process PDF ----------
if uploaded and st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Reading PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded.read())
            path = f.name
        pages = PyPDFLoader(path).load()
        os.remove(path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        vectors = np.array(list(model.embed([c.page_content for c in chunks])), dtype="float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.file_name = uploaded.name
        st.session_state.messages = []

# ---------- Answer function ----------
def answer(question):
    chunks = st.session_state.chunks
    q = np.array(list(model.embed([question])), dtype="float32")
    faiss.normalize_L2(q)
    _, ids = st.session_state.index.search(q, k)

    context = "\n\n".join(chunks[i].page_content for i in ids[0])
    pages_used = sorted(set(chunks[i].metadata["page"] + 1 for i in ids[0]))

    prompt = f"""Answer the question using ONLY the context below.
Explain in simple words. If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question: {question}
"""
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)
    return resp.text, pages_used

# ---------- Tabs ----------
st.title("📚 PDF Q&A Chatbot")
tab_chat, tab_doc, tab_about = st.tabs(["💬 Chat", "📄 Document", "ℹ️ About"])

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "index" not in st.session_state:
        st.info("Upload a PDF from the sidebar to start.")
    elif not api_key:
        st.warning("Add your Gemini API key.")
    else:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m.get("pages"):
                    st.caption(f"Pages searched: {m['pages']}")

        question = st.chat_input("Ask a question about your PDF")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    text, pages_used = answer(question)
                st.write(text)
                st.caption(f"Pages searched: {pages_used}")
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "pages": pages_used}
            )

with tab_doc:
    if "pages" not in st.session_state:
        st.info("No PDF uploaded yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("File", st.session_state.file_name[:20])
        c2.metric("Pages", len(st.session_state.pages))
        c3.metric("Chunks", len(st.session_state.chunks))
        n = st.selectbox("Preview page", range(1, len(st.session_state.pages) + 1))
        st.text(st.session_state.pages[n - 1].page_content)

with tab_about:
    st.markdown("""
**How it works (RAG)**
1. PDF is split into small chunks
2. Chunks become embeddings and are stored in FAISS
3. Your question finds the most related chunks
4. Gemini writes the answer using only those chunks

**Tech:** Python, LangChain, FastEmbed, FAISS, Gemini, Streamlit
""")