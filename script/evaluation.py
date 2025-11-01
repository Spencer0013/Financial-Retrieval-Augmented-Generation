import os
import logging
from itertools import islice
from typing import List, Tuple, Dict, Any

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from helpers import (
    read_queries_jsonl,
    build_rag_chain,
    judge_answer,
    GEN_MODEL,
    JUDGE_MODEL,
)

load_dotenv()

# default
QUERIES_PATH      = "fiqa/queries.jsonl"
BM25_STATE        = r"C:\Users\ainao\dev\Financial-Retrieval-Augmented-Generation\bm25_values.json"
INDEX_NAME        = "fiqa-hybrid"
OUTPUT_CSV        = "eval_results.csv"
FIRST_N           = 100
JUDGE_SAMPLES     = 3
GEN_MODEL_NAME    = GEN_MODEL        
JUDGE_MODEL_NAME  = JUDGE_MODEL      


def evaluate(
    queries_path: str,
    bm25_path: str,
    index_name: str,
    out_csv: str,
    first_n: int,
    gen_model_name: str,
    judge_model_name: str,
    judge_samples: int,
) -> pd.DataFrame:

    # Load queries
    all_queries: List[Tuple[str, str]] = read_queries_jsonl(queries_path)
    queries = list(islice(all_queries, first_n if first_n and first_n > 0 else len(all_queries)))

    # Build retriever and rag chain
    retriever, rag_chain = build_rag_chain(
        index_name=index_name,
        bm25_path=bm25_path,
        gen_model=gen_model_name,
    )

    rows: List[Dict[str, Any]] = []

    for qid, question in tqdm(queries, desc=f"Evaluating {len(queries)} queries", unit="q", dynamic_ncols=True):
        # Get answer from the RAG chain
        rag_out = rag_chain.invoke({"input": question})
        answer = rag_out.get("answer", "")

        # Build context for judging from retriever directly
        docs = retriever.invoke(question)
        context_text = "\n\n".join((getattr(d, "page_content", "") or "") for d in docs)

        # Score with judge
        judged = judge_answer(
            judge_model=judge_model_name,
            question=question,
            context=context_text,
            answer=answer,
            n_samples=judge_samples,
        )

        rows.append({
            "query_id": qid,
            "question": question,
            "answer": answer,
            "correctness": judged["correctness"],
            "groundedness": judged["groundedness"],
            "judge_notes": judged["notes"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY is not set in environment.")
    if not os.getenv("PINECONE_API_KEY"):
        logging.warning("PINECONE_API_KEY is not set in environment.")

    print(" Running FiQA RAG evaluation")
    print(f"- queries_path: {QUERIES_PATH}")
    print(f"- bm25_path:    {BM25_STATE}")
    print(f"- index_name:   {INDEX_NAME}")
    print(f"- out_csv:      {OUTPUT_CSV}")
    print(f"- first_n:      {FIRST_N}")
    print(f"- gen_model:    {GEN_MODEL_NAME}")
    print(f"- judge_model:  {JUDGE_MODEL_NAME}")
    print(f"- judge_samples:{JUDGE_SAMPLES}")

    df = evaluate(
        queries_path=QUERIES_PATH,
        bm25_path=BM25_STATE,
        index_name=INDEX_NAME,
        out_csv=OUTPUT_CSV,
        first_n=FIRST_N,
        gen_model_name=GEN_MODEL_NAME,
        judge_model_name=JUDGE_MODEL_NAME,
        judge_samples=JUDGE_SAMPLES,
    )

    avg_c = float(df["correctness"].mean()) if len(df) else 0.0
    avg_g = float(df["groundedness"].mean()) if len(df) else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Queries Evaluated: {len(df)}")
    print(f"Avg Correctness:   {avg_c:.3f}")
    print(f"Avg Groundedness:  {avg_g:.3f}")
    print(f"Results saved to:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

