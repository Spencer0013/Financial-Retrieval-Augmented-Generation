
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from script.helpers import (
    build_rag_chain,
    GEN_MODEL,
)

load_dotenv(find_dotenv(), override=False)

# Config 
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fiqa-hybrid")
BM25_PATH = os.getenv(
    "BM25_STATE_PATH",
    r"C:\Users\ainao\dev\Financial-Retrieval-Augmented-Generation\bm25_values.json"
)
GEN_MODEL  = GEN_MODEL

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY")
if not os.path.exists(BM25_PATH):
    raise RuntimeError(f"BM25 state not found at {BM25_PATH} (dump it first)")

# Build retriever and RAG chain 
retriever, rag_chain = build_rag_chain(
    index_name=INDEX_NAME,
    bm25_path=BM25_PATH,
    gen_model=GEN_MODEL,
)

#Streamlit UI
st.set_page_config(page_title="FiQA Q/A", layout="centered")
st.title("FiQA Q/A Demo")

with st.form(key="qa_form", clear_on_submit=False):
    question = st.text_input("Ask a financial question:")
    submitted = st.form_submit_button("Get answer")

if submitted and question.strip():
    with st.spinner("Answering..."):
        out = rag_chain.invoke({"input": question})
    st.subheader("Answer")
    st.write(out.get("answer", ""))
