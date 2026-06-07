"""Builds a dual-retrieval (FAISS + BM25) search index for CMS query classification."""

import json
import re
import string
import os
import time
from pathlib import Path

import faiss
import joblib
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

CSV_PATH = Path("synthetic_queries.csv")
ARTIFACTS_DIR = Path("artifacts")
CONFIG_PATH = ARTIFACTS_DIR / "config.json"
HF_TOKEN = os.getenv("HF_TOKEN")

# Default Config for the training and indexing pipeline. 
# This will be overridden by any existing config.json values, and updated with runtime stats before saving. Adjust BM25 params here as needed.
DEFAULT_CONFIG = {
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "embedding_query_instruction": "Represent this sentence: ",
    "faiss_index_type": "IndexFlatIP",
    "num_classes": 50,
    "num_samples": 0,
    "bm25_params": {
        "k1": 1.2,
        "b": 0.75,
        "epsilon": 0.25,
    },
    "top_k_retrieval": 50,
    "rrf_k_constant": 60,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_top_n": 20,
}


def load_config() -> dict:
    """Load config from artifacts/config.json if it exists, else return defaults."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        print(f"Loaded existing config from {CONFIG_PATH}")
    else:
        config = DEFAULT_CONFIG.copy()
        config["bm25_params"] = DEFAULT_CONFIG["bm25_params"].copy()
        print("No existing config found — using defaults")
    return config


def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    """Load synthetic_queries.csv, validate, encode labels, save LabelEncoder."""
    df = pd.read_csv(csv_path, dtype=str)
    df["query"] = df["query"].str.strip()
    df["target_class"] = df["target_class"].str.strip()

    null_counts = df[["query", "target_class"]].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

    print(f"\nClass distribution ({df['target_class'].nunique()} classes):")
    print(df["target_class"].value_counts().to_string())

    le = LabelEncoder()
    df["class_id"] = le.fit_transform(df["target_class"])

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.pkl")
    print(f"\nLabelEncoder saved → {ARTIFACTS_DIR / 'label_encoder.pkl'}")

    return df


def build_faiss_index(df: pd.DataFrame, config: dict) -> None:
    """Encode corpus queries, L2-normalize them, build an IndexFlatIP FAISS index, and save metadata."""
    model = SentenceTransformer(config["embedding_model"])

    embeddings = model.encode(
        df["query"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(ARTIFACTS_DIR / "faiss_index.index"))
    print(f"FAISS index saved → {ARTIFACTS_DIR / 'faiss_index.index'}")

    metadata = [
        {"query": row["query"], "class": row["target_class"], "class_id": int(row["class_id"])}
        for _, row in df.iterrows()
    ]
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved → {ARTIFACTS_DIR / 'metadata.json'}")

    config["num_samples"] = len(df)
    config["num_classes"] = df["target_class"].nunique()
    config["embedding_dim"] = embeddings.shape[1]
    config["faiss_index_size"] = index.ntotal


def build_bm25_index(df: pd.DataFrame, config: dict) -> None:
    """Tokenize queries with punctuation stripping and custom stopword removal, then fit BM25Okapi."""
    bm25_params = config["bm25_params"]
    k1 = bm25_params["k1"]
    b = bm25_params["b"]
    epsilon = bm25_params["epsilon"]
    # Custom stopwords to remove common non-informative tokens
    CUSTOM_STOPWORDS = {
        "i", "me", "my", "a", "an", "the", "for", "to", "with", "and", "or",
        "need", "want", "looking", "find", "doctor", "provider", "specialist",
        "help", "bad", "good", "see", "care"
    }

    print(f"\nBM25 params: k1={k1}  b={b}  epsilon={epsilon}")

    punct_re = re.compile(f"[{re.escape(string.punctuation)}]")

    def tokenize(text: str) -> list[str]:
        tokens = punct_re.sub(" ", text.lower()).split()
        return [tok for tok in tokens if tok not in CUSTOM_STOPWORDS]

    tokenized = [tokenize(q) for q in df["query"].tolist()]
    bm25 = BM25Okapi(tokenized, k1=k1, b=b, epsilon=epsilon)

    joblib.dump(bm25, ARTIFACTS_DIR / "bm25_index.pkl")
    print(f"BM25 index saved → {ARTIFACTS_DIR / 'bm25_index.pkl'}")

    config["bm25_corpus_size"] = len(tokenized)


def save_config(config: dict) -> None:
    """Write config dict to artifacts/config.json."""
    # Remove runtime-only keys before persisting
    persist = {k: v for k, v in config.items() if k not in ("embedding_dim", "faiss_index_size", "bm25_corpus_size")}
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(persist, f, indent=2)
    print(f"\nConfig saved → {CONFIG_PATH}")


def run_training_pipeline(csv_path: Path = CSV_PATH, progress_callback=None) -> dict:
    """Run the full training/indexing pipeline and return the final config."""
    config = load_config()

    if progress_callback:
        progress_callback("load", "Loading and validating synthetic queries")
    df = load_and_validate_data(csv_path)
    config["num_samples"] = len(df)
    config["num_classes"] = df["target_class"].nunique()
    save_config(config)

    if progress_callback:
        progress_callback("faiss", f"Building dense index with {config['embedding_model']}")
    build_faiss_index(df, config)

    if progress_callback:
        progress_callback("bm25", "Building BM25 sparse index")
    build_bm25_index(df, config)

    save_config(config)
    if progress_callback:
        progress_callback("complete", "Artifacts saved")
    return config


def main() -> None:
    # Step 1 — load & validate
    t0 = time.time()
    print("\n=== Step 1: Load & Validate ===")
    config = load_config()
    df = load_and_validate_data(CSV_PATH)
    print(f"Step 1 done in {time.time() - t0:.1f}s")

    # Writing config file to capture the given configurations and dataset stats before we start the indexing steps. 
    config["num_samples"] = len(df)
    config["num_classes"] = df["target_class"].nunique()
    save_config(config)

    # Step 2 — dense embeddings + FAISS
    t0 = time.time()
    print("\n=== Step 2: Dense Embeddings + FAISS ===")
    build_faiss_index(df, config)
    print(f"Step 2 done in {time.time() - t0:.1f}s")

    # Step 3 — sparse BM25 lexical index
    t0 = time.time()
    print("\n=== Step 3: BM25 Sparse Index ===")
    build_bm25_index(df, config)
    print(f"Step 3 done in {time.time() - t0:.1f}s")

    # Step 4 — persist final config
    print("\n=== Step 4: Save Config ===")
    save_config(config)

    print("\n=== Summary ===")
    print(f"  Total samples        : {config['num_samples']}")
    print(f"  Classes found        : {config['num_classes']}")
    print(f"  Embedding dimension  : {config.get('embedding_dim', 'N/A')}")
    print(f"  FAISS index size     : {config.get('faiss_index_size', 'N/A')} vectors")
    print(f"  BM25 corpus size     : {config.get('bm25_corpus_size', 'N/A')} documents")
    bp = config["bm25_params"]

if __name__ == "__main__":
    main()
