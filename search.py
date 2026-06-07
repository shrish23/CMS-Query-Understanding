"""CMS query classification: dual retrieval → RRF → cross-encoder reranking → class aggregation.
"""

import argparse
import json
import re
import string
import os
import sys
from pathlib import Path
from typing import Any
from collections import defaultdict

import faiss
import joblib
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
HF_TOKEN = os.getenv("HF_TOKEN")


def tokenize(text: str) -> list[str]:
    return PUNCT_RE.sub(" ", text.lower()).split()


def load_artifacts() -> dict[str, Any]:
    """Load retrieval artifacts/config from disk and instantiate the embedding and cross-encoder models."""
    try:
        with open(ARTIFACTS_DIR / "config.json") as f:
            config = json.load(f)
    except FileNotFoundError as e:
        sys.exit(f"Artifact not found: {e}. Run train_and_index.py first.")

    with open(ARTIFACTS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    try:
        faiss_index = faiss.read_index(str(ARTIFACTS_DIR / "faiss_index.index"))
    except Exception as e:
        sys.exit(f"Failed to load FAISS index: {e}")

    bm25 = joblib.load(ARTIFACTS_DIR / "bm25_index.pkl")

    try:
        embed_model = SentenceTransformer(config["embedding_model"], token=HF_TOKEN)
    except Exception as e:
        sys.exit(f"Failed to load embedding model '{config['embedding_model']}': {e}")

    try:
        cross_encoder = CrossEncoder(config["cross_encoder_model"], token=HF_TOKEN)
    except Exception as e:
        sys.exit(f"Failed to load cross-encoder '{config['cross_encoder_model']}': {e}")

    return {
        "config": config,
        "metadata": metadata,
        "faiss_index": faiss_index,
        "bm25": bm25,
        "embed_model": embed_model,
        "cross_encoder": cross_encoder,
    }


def dense_retrieve(
    query: str,
    embed_model: SentenceTransformer,
    faiss_index: faiss.Index,
    metadata: list[dict],
    top_k: int,
    query_instruction: str = "",
) -> list[tuple[int, float]]:
    """Embed the query, L2-normalize it, and search FAISS. Returns [(metadata_idx, cosine_score), ...].

    query_instruction: prepended to the query at encode time only (BGE-style retrieval prefix).
    Corpus documents are indexed without any prefix; only the search query is prefixed.
    """
    query_text = query_instruction + query if query_instruction else query
    vec = embed_model.encode([query_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    distances, indices = faiss_index.search(vec, top_k)
    return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx != -1]


def sparse_retrieve(
    query: str,
    bm25: Any,
    metadata: list[dict],
    top_k: int,
) -> list[tuple[int, float]]:
    """Tokenize query and run BM25. Returns [(metadata_idx, bm25_score), ...]."""
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int,
) -> list[tuple[int, float]]:
    """Merge ranked lists with RRF: score = sum(1 / (k + rank)) across lists."""
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def cross_encoder_rerank(
    query: str,
    candidates: list[tuple[int, float]],
    metadata: list[dict],
    cross_encoder: CrossEncoder,
) -> list[tuple[int, float]]:
    """Score query/candidate pairs with a cross-encoder and return candidate records sorted by CE score."""
    if not candidates:
        return []

    pairs = []
    for idx, _rrf_score in candidates:
        candidate_text = (
            f"Matched query: {metadata[idx]['query']}. "
            f"CMS specialty: {metadata[idx]['class']}."
        )
        pairs.append((query, candidate_text))

    logits = cross_encoder.predict(pairs)
    ce_scores = 1.0 / (1.0 + np.exp(-np.asarray(logits)))

    reranked = []
    for (idx, rrf_score), ce_score in zip(candidates, ce_scores):
        reranked.append(
            {
                "idx": int(idx),
                "rrf_score": float(rrf_score),
                "ce_score": float(ce_score),
            }
        )

    return sorted(reranked, key=lambda x: x["ce_score"], reverse=True)


def add_final_candidate_scores(
    reranked: list[dict[str, Any]],
    ce_weight: float = 0.85,
    rrf_weight: float = 0.15,
) -> list[dict[str, Any]]:
    """Add normalized RRF and blended final scores to reranked candidate records, then sort by final score."""
    if not reranked:
        return []

    rrf_values = np.array([x["rrf_score"] for x in reranked], dtype=np.float32)
    rrf_min = float(rrf_values.min())
    rrf_max = float(rrf_values.max())

    for item in reranked:
        if rrf_max > rrf_min:
            rrf_norm = (item["rrf_score"] - rrf_min) / (rrf_max - rrf_min)
        else:
            rrf_norm = 1.0

        item["rrf_norm"] = float(rrf_norm)
        item["final_score"] = (
            ce_weight * item["ce_score"]
            + rrf_weight * item["rrf_norm"]
        )

    return sorted(reranked, key=lambda x: x["final_score"], reverse=True)


def aggregate_class_scores(
    reranked: list[dict[str, Any]],
    metadata: list[dict],
    top_m: int = 3,
) -> dict[str, float]:
    """Aggregate per-class scores from reranked candidates using max score, mean top-M score, and support bonus."""
    grouped: dict[str, list[float]] = defaultdict(list)

    for item in reranked:
        idx = item["idx"]
        cls = metadata[idx]["class"]
        grouped[cls].append(float(item["final_score"]))

    if not grouped:
        return {}

    max_group_size = max(len(scores) for scores in grouped.values())
    class_scores: dict[str, float] = {}

    for cls, scores in grouped.items():
        scores = sorted(scores, reverse=True)

        max_score = scores[0]
        mean_top_m = float(np.mean(scores[:top_m]))
        support_bonus = min(len(scores) / max_group_size, 1.0)

        # Final class score is a weighted combination of the max score, mean top-M score, and support bonus.
        class_scores[cls] = (
            0.65 * max_score
            + 0.25 * mean_top_m
            + 0.10 * support_bonus
        )

    return class_scores



def top_n_independent_scores(
    scores_dict: dict[str, float],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Return the top N classes with independent rounded relevance scores that do not sum to 1."""
    top_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(cls, round(float(score), 2)) for cls, score in top_items]


def format_output(top3: list[tuple[str, float]]) -> str:
    """Format top-3 as: '1. Class A (0.88) | 2. Class B (0.65) | 3. Class C (0.42)'"""
    parts = [f"{i}. {cls} ({score:.2f})" for i, (cls, score) in enumerate(top3, start=1)]
    return " | ".join(parts)


def predict_query(
    query: str,
    artifacts: dict[str, Any] | None = None,
    score_top_n: int = 3,
) -> dict[str, Any]:
    """Run the full retrieval pipeline for one query and return predictions plus intermediate diagnostics."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty or whitespace-only")

    arts = artifacts or load_artifacts()
    config = arts["config"]
    metadata = arts["metadata"]
    top_k = config["top_k_retrieval"]
    rrf_k = config["rrf_k_constant"]
    rerank_top_n = config["rerank_top_n"]

    dense_hits = dense_retrieve(query, arts["embed_model"], arts["faiss_index"], metadata, top_k)
    sparse_hits = sparse_retrieve(query, arts["bm25"], metadata, top_k)
    fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=rrf_k)
    rrf_candidates = fused[:rerank_top_n]
    reranked = cross_encoder_rerank(query, rrf_candidates, metadata, arts["cross_encoder"])
    reranked = add_final_candidate_scores(reranked)
    class_scores = aggregate_class_scores(reranked, metadata, top_m=3)
    top3 = top_n_independent_scores(class_scores, top_n=3)

    return {
        "query": query,
        "config": config,
        "metadata": metadata,
        "dense_hits": dense_hits,
        "sparse_hits": sparse_hits,
        "rrf_candidates": rrf_candidates,
        "reranked": reranked,
        "class_scores": class_scores,
        "top3": top3,
        "formatted_output": format_output(top3),
        "score_top_n": score_top_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CMS query classifier")
    parser.add_argument("query", nargs="?", help="Free-text patient query")
    parser.add_argument("--verbose", action="store_true", help="Print debug pipeline details")
    args = parser.parse_args()

    if not args.query:
        parser.print_usage()
        sys.exit(1)

    query = args.query.strip()
    if not query:
        sys.exit("Error: query must not be empty or whitespace-only.")
    result = predict_query(query, score_top_n=3)
    metadata = result["metadata"]
    dense_hits = result["dense_hits"]
    sparse_hits = result["sparse_hits"]
    rrf_candidates = result["rrf_candidates"]
    reranked = result["reranked"]
    class_scores = result["class_scores"]

    print(result["formatted_output"])

    if args.verbose:
        print("\n--- Verbose Debug ---")
        print(f"Query: {result['query']!r}")

        print("\nTop 10 FAISS hits:")
        for idx, score in dense_hits[:10]:
            print(f"  [{score:.4f}] {metadata[idx]['class']!r}  ← {metadata[idx]['query']!r}")

        print("\nTop 10 BM25 hits:")
        for idx, score in sparse_hits[:10]:
            print(f"  [{score:.4f}] {metadata[idx]['class']!r}  ← {metadata[idx]['query']!r}")

        print("\nTop 20 post-RRF candidates:")
        for idx, score in rrf_candidates:
            print(f"  [{score:.5f}] {metadata[idx]['class']!r}")

        print("\nTop 20 post-cross-encoder candidates:")
        for idx, score in reranked:
            print(f"  [{score:.4f}] {metadata[idx]['class']!r}  ← {metadata[idx]['query']!r}")

        print("\nClass scores (pre-softmax):")
        for cls, score in sorted(class_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cls}: {score:.4f}")

        print(f"\nFinal Top {result['score_top_n']} class scores:")
        for cls, score in result["top3"]:
            print(f"  {cls}: {score:.4f}")


if __name__ == "__main__":
    main()
