## Financial Retrieval-Augmented Generation (RAG) — Finance Domain

A practical, end-to-end RAG pipeline for finance questions powered by Qdrant (dense, sparse, and RRF hybrid search), FastEmbed embeddings, and a Streamlit app for interactive querying. It includes notebooks for data prep and evaluation plus a lightweight UI that returns the top-ranked passage.

## Demo (Screenshot)
<img width="1918" height="1004" alt="Screenshot 2025-09-29 101201" src="https://github.com/user-attachments/assets/36f6736a-7372-4a75-a340-a2e039f3f264" />


## Highlights

Dense (BGE-small) + Sparse (BM25) + Reciprocal Rank Fusion (RRF)

Local or remote Qdrant support

Reproducible evaluation with provided notebook and CSV

Streamlit app for quick demos 

## Project Structure


├── app.py                         
├── data_preparation.ipynb         
├── Exploration and Validation.ipynb
├── df_eval.csv                   
└── README.md                      

## 1) Start Qdrant

Run Qdrant locally with Docker:

docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

## 2) Create the Collection & Upload Vectors

Load finance corpus / notebook chunks

Compute dense embeddings with FastEmbed (BAAI/bge-small-en-v1.5)

Create a multi-vector (dense + bm25) collection in Qdrant

Upload points with payload (e.g., text, filename/source, etc.)

The notebook is written to produce a collection named fiqa-hybrid to match the app defaults.

## Run the App

export QDRANT_URL="http://localhost:6333"
export QDRANT_COLLECTION="fiqa-hybrid"
export EMBED_MODEL="BAAI/bge-small-en-v1.5"

streamlit run app.py

## Using the App

Type a question (e.g., “What is corporate finance?”).

Choose a Search mode in the sidebar:

Dense — vector search on BGE-small

Sparse (BM25) — keyword matching

Hybrid (RRF) — fuses dense & sparse via Reciprocal Rank Fusion

(Dense) Toggle exact search if desired

(Hybrid) Tune dense_k and sparse_k prefetch sizes

## Configuration
### Qdrant connection
QDRANT_URL="http://localhost:6333"

### Collection name to query
QDRANT_COLLECTION="fiqa-hybrid"

### Dense embedding model used by FastEmbed
EMBED_MODEL="BAAI/bge-small-en-v1.5"

## Retrieval Evaluation

An example run over 500 queries(dev.tsv) @ k=10 produced:

<img width="581" height="423" alt="image" src="https://github.com/user-attachments/assets/f985173d-1c0a-407d-8b6f-9edbd30db189" />



## Rag Evaluation

Evaluator prompt (system + user)

The answer is injected verbatim between <<<ANSWER>>> and <<<END>>> to avoid accidental prompt mixing.

The evaluator is instructed to output JSON only (no prose, no backticks), so downstream parsing is reliable.

## Inputs to the evaluator

For each example:

question: the user query.

context: the retrieved snippets the generator saw (the notebook builds these earlier).

answer: the generator’s output (verbatim).

The context is not pasted directly into the evaluator prompt body shown above; instead, the rubric requires the evaluator to judge grounding against the same context that produced the answer. (Implementation-wise, the notebook wraps or references that context when calling the evaluator model so the judgment is actually conditioned on it.)

## Batch evaluation loop

For each retrieval mode (dense, sparse, rrf) and for each evaluation query:

Retrieve context (K results).

Generate the answer with the RAG model.

Evaluate using JSON judgment.

Log the outcome (question, answer, label, and explanation)

The notebook uses a progress bar (tqdm) and caches intermediate results in memory; you can export to CSV/JSON if desired.

## Metrics & reporting

From the collected judgments the notebook computes:

Relevance rate (accuracy):
#(RELEVANT) / total, per mode and overall.

Breakdown by retrieval mode: dense vs. sparse vs. RRF.

Error slices:

Examples flagged IRRELEVANT with evaluator explanations.

Random samples of RELEVANT cases for manual spot-checks.

## What the outputs mean

A high RELEVANT rate under RRF indicates the hybrid retriever is delivering better grounding than dense-only or sparse-only.

IRRELEVANT examples + explanations help you pinpoint:

missing/weak context,

generator drift (generic answers),

or contradictions with retrieved facts.

## How It Works 

1. Chunking & Embedding

Split finance texts / notebooks into passages

Generate dense embeddings with BAAI/bge-small-en-v1.5 via FastEmbed

2. Indexing

Store dense vectors and payloads in Qdrant

Enable bm25 for sparse retrieval on the same collection

3. Querying

Dense: cosine/L2 over normalized dense vectors

Sparse: bm25

Hybrid: RRF merges two ranked lists (configurable prefetch sizes)

4. UI

Streamlit app queries Qdrant and renders the top-1 passage with source

Toggle modes and parameters from the sidebar

## Acknowledgements

Qdrant for vector database & BM25 hybrid support

FastEmbed for blazing-fast embedding inference

BAAI/bge-small-en-v1.5 for strong retrieval performance

## License

Add your preferred license (e.g., MIT) to this repository.
