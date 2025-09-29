## Financial Retrieval-Augmented Generation (RAG) — Finance Domain

A practical, end-to-end RAG pipeline for finance questions powered by Qdrant (dense, sparse, and RRF hybrid search), FastEmbed embeddings, and a Streamlit app for interactive querying. It includes notebooks for data prep and evaluation plus a lightweight UI that returns the top-ranked passage.

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
