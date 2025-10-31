## Financial Retrieval-Augmented Generation (RAG) — Finance Domain

A practical, end-to-end RAG pipeline for finance questions powered by **Pinecone hybrid search (dense + sparse)**, `sentence-transformers` embeddings, and a Streamlit app for interactive querying. It includes notebooks for data prep and evaluation plus a lightweight UI that returns an answer grounded in the top-ranked passage.

## Demo (Screenshot)
<img width="1858" height="1010" alt="Screenshot 2025-10-31 110219" src="https://github.com/user-attachments/assets/f59549eb-ce46-4b26-a2c0-8175b0aa464d" />


---

## Highlights

- Dense (`sentence-transformers/all-MiniLM-L6-v2`) + Sparse (BM25 via `pinecone-text`) + Hybrid in Pinecone  
- Pinecone serverless index for storage and retrieval  
- Reproducible evaluation with provided notebook and CSV  
- Streamlit app for quick demos  
- RAG chain that constrains the model to only use retrieved context (and say “I don’t know” if the answer is not supported)

---

## Project Structure

```text
├── app.py                        # Streamlit application for interactive Q/A
├── helpers.py                    # Retrieval + RAG chain builder, judge prompt
├── evaluation.py                 # Batch evaluation script
├── data_preparation.ipynb        # Data cleaning / chunking / indexing workflow
├── hybrid_search.ipynb           # Retrieval experiments and diagnostics
├── bm25_values.json              # Saved BM25 state (sparse encoder stats)
├── eval_results.csv              # Example evaluation output
├── requirements.txt              # Dependencies
└── README.md
```

> **Note:** earlier versions referenced Qdrant. The current pipeline uses **Pinecone** as the vector DB and hybrid retriever.

---

## 1️ Start / Prepare Pinecone

You do not run Pinecone locally with Docker — it’s a managed service.  
Instead:

1. Create a Pinecone account / API key.  
2. Set environment variables (see below).  
3. The code will create (or reuse) a serverless Pinecone index with the expected name (default: `fiqa-hybrid`).

**Index details (from `helpers.py`):**
- Dimension: `384`
- Metric: `dotproduct`
- Cloud/region: `aws` / `us-east-1` (can be changed in code)

---

##  Create the Index & Upload Vectors

The workflow in `data_preparation.ipynb` / ingestion script does the following:

- Load finance corpus / notebook chunks  
- Fit a BM25 encoder:
  ```python
  from pinecone_text.sparse import BM25Encoder
  bm25 = BM25Encoder().default()
  bm25.fit(texts)
  bm25.dump("bm25_values.json")
  ```
- Compute dense embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Upsert **both** dense and sparse vectors into the Pinecone index along with metadata

>  After you upsert once, your vectors live in Pinecone permanently.  
> You do **not** need to re-upsert unless you add new data.  
> Keep the `bm25_values.json` file — it’s required at query time to encode new user queries into sparse form.

---

##  Run the App

Set your environment variables in a `.env` file or via shell.  
**Required:**

```bash
OPENAI_API_KEY="your OpenAI API key"
PINECONE_API_KEY="your Pinecone API key"
PINECONE_INDEX_NAME="fiqa-hybrid"
BM25_STATE_PATH="C:\path\to\bm25_values.json"
```

Then launch:

```bash
streamlit run app.py
```

### What `app.py` does
- Loads env vars  
- Loads the saved BM25 state (`bm25_values.json`)  
- Connects to the existing Pinecone index (`fiqa-hybrid`)  
- Builds a RAG chain with:
  - `PineconeHybridSearchRetriever` (dense + sparse hybrid search)
  - `gpt-4o-mini` as the generator  
- Lets you ask finance questions and returns grounded answers

---

##  Using the App

1. Type a question (e.g., “What is corporate finance?”).  
2. Submit.  
3. The app:
   - Retrieves top-matching passages from Pinecone using hybrid search  
   - Builds context for the LLM  
   - Calls the generator model  
   - Displays the final grounded answer  

If the answer is not supported by the retrieved context, the model will respond: **“I don’t know.”**

---

## ⚙️ Configuration

### Pinecone connection
- `PINECONE_API_KEY` must be set  
- `PINECONE_INDEX_NAME` defaults to `fiqa-hybrid`  
- The code will create the index automatically if it doesn’t exist (serverless spec)

### BM25 state
- `BM25_STATE_PATH` must point to `bm25_values.json`  
- This file is produced once during ingestion  
- Without it, retrieval fails with:  
  `RuntimeError: BM25 state not found ... (dump it first)`

### Generator model
- Uses `gpt-4o-mini` via `langchain-openai`  

### Embedding model
- Dense encoder: `sentence-transformers/all-MiniLM-L6-v2`

---

##  Retrieval Evaluation

You can batch-score the system with:

```bash
python evaluation.py
```

This script:
- Loads a list of test queries (`fiqa/queries.jsonl`)
- Builds the same retriever + RAG chain as the app
- Generates answers and evaluates them using a judge model (`gpt-4o-mini`)
- Computes:
  - **correctness** — does the answer address the question accurately and completely?  
  - **groundedness** — is it supported only by retrieved context?

Results are saved to **`eval_results.csv`**.

### Example output columns
| Field | Description |
|--------|--------------|
| question | The query asked |
| answer | Model-generated response |
| correctness | Score (0–1) |
| groundedness | Score (0–1) |
| judge_notes | Short textual rationale |

---

##  Metrics & Reporting

From the evaluation CSV you can compute:

- Overall relevance / correctness rate  
- Groundedness (hallucination control)  
- Breakdowns by retrieval configuration  
- Slices of low-performing examples for manual inspection  

Higher groundedness = stronger retrieval-grounded responses.

---

##  How It Works (End-to-End)

1. **Chunking & Embedding**  
   Split finance data into passages → create dense embeddings and BM25 sparse vectors.  

2. **Indexing**  
   Create Pinecone index → upsert dense + sparse + metadata.  

3. **Querying**  
   Load BM25 + dense encoder → perform hybrid retrieval via `PineconeHybridSearchRetriever`.  

4. **Answering**  
   Retrieved docs → passed as `context` to `gpt-4o-mini` → returns a concise, grounded answer.  

5. **Evaluation**  
   Use `evaluation.py` to assess correctness & groundedness with the same LLM as a judge.

---

## Acknowledgements

- **Pinecone** for hybrid dense/sparse retrieval and vector storage  
- **pinecone-text** for BM25Encoder  
- **sentence-transformers** for MiniLM embeddings  
- **LangChain** for retrieval + RAG orchestration  
- **OpenAI gpt-4o-mini** for answering and judging  

---

## 🪪 License

Add your preferred license (e.g., MIT) to this repository.
