import os
import json
import textwrap
from typing import List, Tuple, Dict, Any
from collections import Counter

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "True")
os.environ.setdefault("LANGCHAIN_API_KEY", "")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "")

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain



HF_EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2" 
GEN_MODEL        = "gpt-4o-mini"
JUDGE_MODEL      = "gpt-4o-mini"
PINECONE_CLOUD   = "aws"
PINECONE_REGION  = "us-east-1"


# Small file helpers
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out


def read_queries_jsonl(path: str) -> List[Tuple[str, str]]:
    data = read_jsonl(path)
    pairs: List[Tuple[str, str]] = []
    for obj in data:
        qid = str(obj.get("_id"))
        text = (obj.get("text") or "").strip()
        if qid and text:
            pairs.append((qid, text))
    if not pairs:
        raise ValueError(f"No queries found in {path}")
    return pairs


# Pinecone index helper
def ensure_pinecone_index(
    pc: Pinecone,
    index_name: str,
    dimension: int = 384,
    metric: str = "dotproduct",
    cloud: str = PINECONE_CLOUD,
    region: str = PINECONE_REGION,
) -> None:

    existing = {ix["name"] for ix in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )


# Core builder for retriever and rag_chain
def build_rag_chain(
    index_name: str,
    bm25_path: str,
    gen_model: str = GEN_MODEL,
):
   
    # Check env
    openai_key   = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    if not pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY")

    # Pinecone setup
    pc = Pinecone(api_key=pinecone_key)
    ensure_pinecone_index(pc, index_name=index_name, dimension=384, metric="dotproduct")
    index = pc.Index(index_name)

    # Hybrid retriever components
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    bm25 = BM25Encoder().load(bm25_path)

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
    )

    # LLM and prompt
    llm = ChatOpenAI(model=gen_model, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant. Answer strictly using the provided context. "
             "If the answer is not in the context, say: \"I don't know\". Be concise."),
            (
                "human",
                "Answer the following question based only on the provided context.\n"
                "Think step by step before providing a detailed answer.\n"
                "<context>\n{context}\n</context>\n"
                "Question: {input}"
            ),
        ]
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return retriever, rag_chain


# LLM-as-Judge prompt and scoring
JUDGE_PROMPT = ChatPromptTemplate.from_template(
    textwrap.dedent("""\
    You are a strict evaluator for Retrieval-Augmented Generation (RAG).
    Given a QUESTION, CONTEXT, and ANSWER, return pure JSON with scores in [0,1]:
    {{"correctness": <float>, "groundedness": <float>, "notes": "<short reason>"}}

    Definitions:
    - correctness: does the answer address the question accurately and completely?
    - groundedness: are the claims supported ONLY by the provided context (no hallucinations)?

    QUESTION: {question}

    CONTEXT:
    {context}

    ANSWER:
    {answer}
    """)
)


def _judge_once(
    judge_llm: ChatOpenAI,
    question: str,
    context: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Call judge model once and parse the strict JSON.
    Fallback: (0.0, 0.0, "parse-fail") if we can't parse.
    """
    msg = JUDGE_PROMPT.invoke({"question": question, "context": context, "answer": answer})
    resp = judge_llm.invoke(msg)
    raw = (getattr(resp, "content", None) or str(resp)).strip()


    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw

    try:
        obj = json.loads(raw)
        correctness  = float(obj.get("correctness", 0.0))
        groundedness = float(obj.get("groundedness", 0.0))
        notes        = str(obj.get("notes", ""))[:400]
    except Exception:
        correctness, groundedness, notes = 0.0, 0.0, "parse-fail"

    return {
        "correctness":  correctness,
        "groundedness": groundedness,
        "notes":        notes,
        "raw":          raw,
    }


def judge_answer(
    judge_model: str,
    question: str,
    context: str,
    answer: str,
    n_samples: int = 3,
) -> Dict[str, Any]:
    """
    Self-consistency ensemble judge:
      - Calls LLM multiple times
      - Averages correctness/groundedness
      - Picks the most common note
    """
    judge_llm = ChatOpenAI(model=judge_model, temperature=0)

    corr_scores, ground_scores, notes_list, raws = [], [], [], []

    for _ in range(max(1, n_samples)):
        out = _judge_once(judge_llm, question, context, answer)
        corr_scores.append(out["correctness"])
        ground_scores.append(out["groundedness"])
        notes_list.append(out["notes"])
        raws.append(out["raw"])

    avg_corr = sum(corr_scores) / len(corr_scores)
    avg_ground = sum(ground_scores) / len(ground_scores)
    common_note = Counter(notes_list).most_common(1)[0][0] if notes_list else ""

    return {
        "correctness":  round(avg_corr, 3),
        "groundedness": round(avg_ground, 3),
        "notes":        common_note,
        "samples": [
            {
                "correctness":  c,
                "groundedness": g,
                "notes":        n,
            }
            for c, g, n in zip(corr_scores, ground_scores, notes_list)
        ],
        "raws": raws,
    }
