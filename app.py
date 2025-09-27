# app.py

import os
import math
from typing import List
import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# Config (override via env vars)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "fiqa-hybrid")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


# Init

@st.cache_resource(show_spinner=False)
def get_client() -> QdrantClient:
    return QdrantClient(QDRANT_URL)

@st.cache_resource(show_spinner=False)
def get_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL)

client = get_client()
embedder = get_embedder()


# Helper functions
def _l2(v):
    if hasattr(v, "tolist"):
        v = v.tolist()
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [float(x / n) for x in v]

def embed_query(text: str) -> List[float]:
    s = f"Represent this sentence for retrieval: {text}"
    vec = list(embedder.embed([s]))[0]
    if isinstance(vec, dict) and "embedding" in vec:
        vec = vec["embedding"]
    return _l2(vec)

def top1_dense(query: str, exact: bool = True):
    qvec = embed_query(query)
    res = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        using="dense",
        with_payload=True,
        limit=1, 
        search_params=models.SearchParams(exact=exact),
    )
    return res.points[0] if res.points else None

def top1_sparse(query: str):
    res = client.query_points(
        collection_name=COLLECTION,
        query=models.Document(text=query, model="qdrant/bm25"),
        using="bm25",
        with_payload=True,
        limit=1,  
    )
    return res.points[0] if res.points else None

def top1_rrf(query: str, dense_k: int = 200, sparse_k: int = 400):
    qvec = embed_query(query)
    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=qvec, using="dense", limit=dense_k),
            models.Prefetch(
                query=models.Document(text=query, model="qdrant/bm25"),
                using="bm25",
                limit=sparse_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=1,  # 
    )
    return res.points[0] if res.points else None


# UI
st.set_page_config(page_title="Notebook RAG Search (Top-1)", page_icon="🔎", layout="wide")
st.title(" Finance Domain Question and Answer")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Search mode", ["Dense", "Sparse (BM25)", "Hybrid (RRF)"])
    dense_exact = st.toggle("Dense exact", value=True, help="SearchParams(exact=True)")
    if mode == "Hybrid (RRF)":
        dense_k = st.slider("Dense prefetch (RRF)", 50, 1000, 200, 50)
        sparse_k = st.slider("Sparse prefetch (RRF)", 50, 1000, 400, 50)
    else:
        dense_k, sparse_k = 200, 400
    st.caption(f"Qdrant: {QDRANT_URL}  •  Collection: {COLLECTION}")
    st.caption(f"Embedding: {EMBED_MODEL}")

query = st.text_input("Enter your question", placeholder="e.g., What is corporate finance?")
go = st.button("Search")

def render_best(point: models.ScoredPoint | None):
    if not point:
        st.info("No result found.")
        return
    payload = point.payload or {}
    text = payload.get("text") or ""
    src = payload.get("filename") or payload.get("source") or payload.get("id") or "-"
    score = getattr(point, "score", None)

    st.subheader("Best answer")
    with st.container(border=True):
        st.caption(f"Source: `{src}`" + (f" • score: {score:.4f}" if score is not None else ""))
        # show as body text (not a long code block) for readability
        st.write(text if len(text) <= 4000 else text[:4000] + "…")

if go and query.strip():
    with st.spinner("Searching…"):
        if mode == "Dense":
            best = top1_dense(query, exact=dense_exact)
        elif mode == "Sparse (BM25)":
            best = top1_sparse(query)
        else:
            best = top1_rrf(query, dense_k=dense_k, sparse_k=sparse_k)
    render_best(best)

st.markdown("---")
st.caption("Returns only the top-ranked passage from your Qdrant collection (dense/sparse/RRF).")
