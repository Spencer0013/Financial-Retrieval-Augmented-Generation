## Problem Statement

Financial Q&A is unforgiving: questions often hinge on precise tokens (tickers, ratios, named entities) as well as semantic understanding (definitions, paraphrases, relationships). Pure dense retrieval can miss exact terms; pure lexical search can miss meaning. Large, end-to-end LLM answers are not reliably grounded and may hallucinate without citations.

This capstone delivers a compact, explainable Retrieval-Augmented Generation (RAG) baseline for the FIQA domain that prioritizes trustworthiness, transparency, and reproducibility. It retrieves with dense, sparse (BM25), or hybrid (RRF) search; shows the exact supporting passage(s); and is structured so each component can be audited and upgraded.

## What This Repository Contains

Data Preparation notebook — shapes FIQA data into a clean schema and builds a dual-index (dense + BM25) collection in Qdrant.

Exploration & Validation notebook — sanity-checks retrieval behavior, comparing Dense vs BM25 vs Hybrid (RRF) and documenting findings.

Streamlit application (app.py) — an interactive probe for top-1 retrieval with configurable modes (Dense / BM25 / Hybrid), query embedding, RRF fusion, and explicit provenance display. 

app

This separation makes the project easy to grade: data prep → validation → app.

## System Goals & Design Principles

Accuracy through complementarity: combine semantic recall (dense) with lexical precision (BM25).

Grounded outputs: always show the source passage, score (when available), and collection metadata to keep the loop auditable. 

app

Transparency over scale: default to small, CPU-friendly components; every step is reproducible and swappable.

Separation of concerns: data shaping → indexing → retrieval → fusion → (optional) generation. The app intentionally focuses on retrieval quality before adding an answer generator.

## End-to-End Flow

Data Preparation notebook
Goal: Convert raw FIQA into minimal, self-contained passages and build a dual-index collection.
What it does:

Schema normalization to a compact payload (_id, title, text). Titles are derived from the first sentence if missing.

Dense embedding preparation using a small, retrieval-tuned model; vectors are L2-normalized to align with cosine scoring in Qdrant.

Sparse (BM25) preparation so rare tokens and exact financial terms are captured via an inverted index with IDF weighting.

Collection creation in Qdrant with both a dense vector field and a BM25 sparse field; idempotent (re-creates cleanly).

Upsert of all items with payloads and both representations (dense vector + BM25 document).

Basic QA checks: counts match, spot-check payloads, confirm vector norms, try a few queries to ensure dense/sparse both behave sensibly.

Exploration & Validation notebook
Goal: Verify retrieval quality and that hybrid (RRF) improves robustness.
What it does:

Runs paired queries across Dense, BM25, and Hybrid modes.

Logs qualitative differences: Dense tends to find paraphrases and concept matches; BM25 nails tickers/ratios/entities; Hybrid stabilizes mixed cases.

Notes Top-K behavior and first-hit quality; records edge cases where the corpus lacks coverage; produces a short retrospective on tuning levers (e.g., Top-K, BM25 tweaks, embedding prompt).

Confirms provenance visibility (title/source) for auditability.

Interactive app (app.py)
Goal: Provide a clean, trustworthy interface to test top-1 retrieval modes quickly.
What it does (key points):

Embeds the user’s query with a retrieval prompt (“Represent this sentence for retrieval: …”) and L2-normalizes the vector before dense search. 

Lets you choose Dense, Sparse (BM25), or Hybrid (RRF) from the sidebar. Shows the active Qdrant URL, collection, and embedding model for transparency. 

Dense mode includes a “Dense exact” toggle that forwards exact search params to Qdrant. 

Hybrid (RRF) runs server-side fusion over two prefetch lists (dense and sparse). You can adjust Dense prefetch and Sparse prefetch sliders to explore the effect on fusion; defaults are balanced for demo scale. 


Returns the top-ranked passage, prints source (e.g., filename, source, or id from the payload), and shows the score when available. It trims very long texts for readability. 

Keeps the app responsive via resource caching for the Qdrant client and embedder. 


Purposefully does not generate final natural-language answers: this is a retrieval probe so graders can evaluate ranking quality and evidence integrity before layering generation. The footer explicitly notes “Returns only the top-ranked passage …”. 

## How the Parts Connect

Data Preparation → Qdrant: The notebook creates a named collection (default: fiqa-hybrid) and upserts both dense vectors and BM25 docs.

App reads the same collection: By default, app.py targets the fiqa-hybrid collection at QDRANT_URL (default http://localhost:6333). You can override URL, collection, and model via environment variables. 

Exploration notebook informs app defaults: Findings from validation (e.g., prefetch sizes, dense “exact” toggle) explain the app’s balanced defaults and UI controls. 

## Usage 

## Prepare the data (Data Preparation notebook)

Load FIQA and shape records to the minimal payload (_id, title, text).

Create or re-create the Qdrant collection with both dense and BM25 fields.

Embed passages, normalize vectors, and upsert with payloads + BM25 docs.

Run quick QA checks (counts, payload spot-checks, test queries).

## Explore & validate (Exploration & Validation notebook)

Run the same query across Dense, BM25, and Hybrid.

Record observations where dense helps (paraphrases) vs BM25 (exact tokens).

Confirm Hybrid (RRF) stabilizes rankings and improves first-hit quality.

Capture a short write-up with examples and any tuning recommendations.

## Run the app (app.py)

Confirm QDRANT_URL, collection name, and embedding model through your environment (the app displays them in the sidebar). 

Choose a Search mode and, if Dense, decide whether Dense exact should be on. For Hybrid, pick Dense/Sparse prefetch sizes. 

Enter a question and hit Search.

Review the top passage, its source, and score. If it’s off, try switching modes or tweaking the prefetch sizes to understand why.

## Evaluation Criteria (for a 10/10 README & Capstone)

Relevance & robustness: Hybrid retrieval should handle both keywords and meaning, with stable top-1 behavior across paraphrases.

Transparency: The app shows exact source text, source identifier, score, and collection/model metadata—no hidden steps. 

Reproducibility: Environment configuration is explicit; data prep can recreate the collection deterministically; the app only reads it. 

Methodological clarity: The notebooks explain the why of dense vs sparse vs hybrid, document experiments, and make tuning suggestions grounded in observed behavior.

Ethics & provenance: Sources are surfaced directly; any sensitive content (if encountered) is excluded/redacted during prep.

## What “Good” Looks Like (Reality Check)

Dense returns semantically relevant passages for definition-style queries.

BM25 excels at rare tokens (tickers, ratios) and entity-heavy questions.

Hybrid (RRF) consistently puts an actually useful passage at rank-1 even when dense and sparse disagree.

Reviewers can trace any answer back to the supporting text and re-run the same query reliably.

## Design Choices & Rationale

Hybrid-first: Dense ≈ semantic recall; BM25 ≈ lexical precision. RRF keeps fusion stable without brittle score calibration. The app uses server-side RRF with tunable prefetch lists. 

Small, fast components: A retrieval-tuned small embedder and BM25 index keep ingestion and queries CPU-friendly and demo-ready.

L2 + cosine: Embeddings are normalized to match the collection’s cosine distance; the app normalizes query vectors before search. 

Evidence-first UI: Top passage with source + score encourages disciplined evaluation before adding generation. 

## Limitations (Intentional for Baseline Clarity)

Top-1 only in the app: by design, to focus grading on rank quality and evidence; Top-K display is a straightforward extension. 

No LLM generator in the app: prevents conflating retrieval with model behavior. Add generation after retrieval is validated.

Single-pass retrieval: no iterative query rewriting yet.

No advanced chunking: very long documents may require chunking/overlap in a follow-up.