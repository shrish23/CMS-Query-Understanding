# CMS Query Understanding

This project maps short, free-text patient queries to the most relevant CMS Medicare Provider/Supplier specialties.

It uses a hybrid retrieval pipeline:

- Dense semantic search with `sentence-transformers`
- Sparse BM25 retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- Class-level score aggregation for the final top-3 specialties

The repo also includes a Streamlit app for interactive search, synthetic query generation, and index building.

## Why this approach

Short patient queries are often vague, colloquial, or ambiguous. A closed-set classifier can struggle with that kind of input, especially when the training data is synthetic. Retrieval-based classification is a better fit because it can:

- generalize to unseen phrasing
- surface multiple plausible specialties
- avoid retraining when taxonomy entries change
- combine exact-term and semantic matching

For a deeper rationale and evaluation notes, see [architecture.md](architecture.md).

## Project Structure

```text
.
├── data/
│   └── Medicare_Provider_and_Supplier_Taxonomy_Crosswalk_October_2025.csv
├── generate_data.py
├── synthetic_queries.csv
├── train_and_index.py
├── artifacts/
│   ├── bm25_index.pkl
│   ├── faiss_index.index
│   ├── label_encoder.pkl
│   ├── metadata.json
│   └── config.json
├── search.py
├── streamlit_app.py
├── architecture.md
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- A Hugging Face token in `HF_TOKEN` if you want to generate synthetic data or download gated models

## Install

```bash
pip install -r requirements.txt
```

If you plan to generate data, set your Hugging Face token first:

```bash
export HF_TOKEN=your_token_here
```

## Usage

### 1. Generate synthetic queries

This creates `synthetic_queries.csv` from the CMS taxonomy crosswalk.

```bash
python generate_data.py
```

Defaults:

- generation model: `Qwen/Qwen3-4B`
- queries per specialty: `30`

### 2. Build retrieval artifacts

This encodes the synthetic queries, builds the FAISS index, and fits BM25.

```bash
python train_and_index.py
```

Outputs are written to `artifacts/`:

- `faiss_index.index`
- `bm25_index.pkl`
- `metadata.json`
- `label_encoder.pkl`
- `config.json`

### 3. Run a single query from the CLI

```bash
python search.py "bad back"
```

Add `--verbose` to inspect the full retrieval pipeline:

```bash
python search.py "bad back" --verbose
```

### 4. Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app includes:

- query search
- synthetic data generation
- artifact building

## How It Works

1. The query is embedded with `BAAI/bge-large-en-v1.5`.
2. The query is also tokenized for BM25.
3. Dense and sparse results are fused with Reciprocal Rank Fusion.
4. The top candidates are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
5. Candidate scores are aggregated to produce the final top-3 CMS specialties.

## Evaluation

The architecture file includes a holdout comparison between:

- Retrieval + embeddings
- TF-IDF + Logistic Regression

The retrieval-based approach performs better across the reported ranking and classification metrics.

## Notes

- The project is designed around synthetic training data and a CMS taxonomy crosswalk.
- The generated artifacts are enough to run inference locally once they exist.
- If you change the synthetic dataset, rerun `train_and_index.py` to rebuild the index.

