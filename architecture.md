# Architecture: CMS Query Understanding System

**Summary:** A hybrid retrieval + cross-encoder rerank pipeline that maps free-text colloquial patient queries to the top-3 official CMS Medicare Provider/Supplier Type Descriptions, with confidence scores.

---

## Q1 — Why this ML approach (Classifier vs. Embeddings vs. NER)?

### Decision: Hybrid Retrieval + Cross-Encoder Rerank

The implemented pipeline combines **dense semantic search** (`BAAI/bge-large-en-v1.5` + FAISS), **sparse BM25 retrieval**, **Reciprocal Rank Fusion**, and a **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

### Why not a supervised multi-class classifier?

- **Data starvation**: only ~30 synthetic training examples per class. A softmax classifier (even fine-tuned BERT) would overfit the synthetic phrasing and fail on the colloquial distribution of real user queries.
- **Brittle to vocabulary shift**: classifier logits are calibrated to training tokens. "Bad back", "hurts when I breathe", "can't sleep" are OOV relative to clinical training corpora — embeddings handle synonymy; a classifier assigns near-zero probability.
- **Poor at ambiguous queries**: queries like "pain", "feeling off", or "need help" do not map cleanly to a single class. A classifier is forced to pick one label, while retrieval can return several plausible specialties with scores.
- **No live labels**: all 1,500 training examples are LLM-generated. There is no real user traffic to validate class boundaries. A retrieval system degrades gracefully; a classifier silently returns overconfident wrong predictions.
- **Taxonomy maintenance cost**: the CMS taxonomy updates annually. Adding or renaming a specialty forces **full retraining + re-evaluation** of a classifier. With retrieval, a taxonomy update = add the new specialty's examples, re-embed, done — no gradient descent required.

### Why not NER + entity linking?

- **Colloquial queries have no named entities**: "I can't stop coughing", "my knee gives out", "numbness in fingers" contain no clinical entity spans to tag. A NER model would find nothing to link.
- **UMLS/SNOMED dependency**: entity linking requires a large medical ontology (UMLS, SNOMED CT) plus a trained linker. This adds ~2 GB of data, a multi-stage pipeline, and a domain-specific model — all for a 50-class problem that simpler retrieval solves directly.
- **Brittle multi-step error propagation**: NER errors compound at the linking stage. End-to-end accuracy of a 3-stage pipeline (raw text → NER → linking → specialty) is the product of each stage's accuracy.

### Why embeddings + retrieval wins here

- **Zero-shot generalization**: `bge-large-en-v1.5` maps "hurts when I breathe" and "chest pain on inhale" to nearly identical embedding vectors, even though the training corpus never saw either phrasing. No retraining required.
- **Taxonomy agility**: corpus updates are offline operations (`python train_and_index.py`). The FAISS index is rebuilt in minutes; inference code is unchanged.
- **Sub-millisecond retrieval**: FAISS `IndexFlatIP` on 1,500 L2-normalized vectors returns top-50 results in <1ms. Cost scales with index size, not class count.
- **Semantic synonym coverage**: embedding similarity handles lay-to-clinical synonymy ("heart doctor" ≈ "Cardiovascular Disease (Cardiology)") without a lexicon.

### Why hybrid retrieval + rerank, not plain embeddings?

- **BM25 catches what dense misses**: exact-token queries ("CRNA anesthesia", "PT shoulder") score near-zero in embedding space if the phrasing is rare in the pre-training corpus. BM25 recovers them via exact term overlap.
- **RRF fusion is calibration-free**: Reciprocal Rank Fusion merges the dense and sparse ranked lists using only rank positions (`1 / (k + rank)`), eliminating the need to calibrate incompatible score scales.
- **Cross-encoder precision**: the bi-encoder produces approximate semantic similarity. The cross-encoder (`ms-marco-MiniLM-L-6-v2`) attends jointly to query and candidate, resolving subtle distinctions (e.g., "spine pain" → Orthopedic Surgery vs. Neurosurgery) that the bi-encoder conflates.
- **Max-pool class aggregation**: scoring at the class level via `max(CE score)` rather than sum avoids artificially rewarding classes with more training examples. Combined with a mean-top-M and support-bonus term, the final score is robust to outlier candidates.

## Evaluation of retrieval + embeddings approach and classifier (TF-IDF + Logistic Regression) on holdout set

- The holdout set was generated with a different LLM, so this comparison reflects how well each approach generalizes to unseen synthetic phrasing rather than memorizing the training generator.
- Retrieval + embeddings outperforms TF-IDF + Logistic Regression (https://github.com/shrish23/CMS-Query-Classifier) across every metric in the table, with the largest gap in Top-3 Accuracy, which is the most relevant measure for a ranked specialty recommendation system.

| System | Top-1 Accuracy | Top-3 Accuracy | MRR@3 | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| Retrieval + Embeddings | 0.4270 | 0.6800 | 0.5383 | 0.4193 | 0.4193 |
| TF-IDF + LogisticRegression | 0.3050 | 0.4540 | 0.3712 | 0.2973 | 0.2973 |



### Acknowledged tradeoffs

- Retrieval quality is capped by synthetic corpus quality. LLM-generated queries may not fully capture real user vocabulary — real search logs would substantially improve recall.
- The cross-encoder adds ~300–600ms on CPU. Acceptable for a demo CLI; requires architectural changes at production scale (see Q2).
- No held-out evaluation set exists (all 1,500 examples are indexed). Accuracy is unverifiable without real labeled queries or a train/test split.

---

## Q2 — Production infrastructure for 5,000 req/s at sub-50ms latency

### Latency budget (50ms target)

| Stage | Dev (CPU) | Target (GPU/optimized) |
|---|---|---|
| Query embedding (bge-large) | ~80–150ms | **≤10ms** (GPU + ONNX-INT8) |
| ANN retrieval (FAISS) | <5ms | **≤2ms** (HNSW / IVF-PQ) |
| BM25 scoring | <10ms | **≤5ms** (in-memory, fast path) |
| Cross-encoder rerank (50 cands) | ~300–600ms | **≤20ms** (distilled, GPU batch) |
| Aggregation + formatting | <1ms | ≤1ms |
| **Total** | **~400–800ms** | **≤38ms** (headroom for I/O) |

### 1. Cache layer (highest impact, lowest cost)

- **Normalized query → result cache** in Redis or Memcached. Autocomplete queries are highly repetitive ("bad bac", "bad back", "back pain" all appear for the same user session).
- **Cache key**: lowercase + punct-stripped query string (same normalization as the `tokenize()`). TTL: 24h.
- **Expected cache hit rate**: 60–80%+ on production autocomplete traffic. This alone eliminates the majority of model calls.
- **Prefix/partial caching**: for autocomplete, cache results for each committed prefix and serve them client-side until the user pauses typing.

### 2. Model serving and optimization

- **Query encoder** (`bge-large-en-v1.5`):
  - Using a **quatized model to INT8** reduces inference from ~100ms CPU to ~8ms GPU, ~20ms on CPU.
  - Alternatively, **distill to `bge-small-en-v1.5`** (33M params, 384-dim); 3× faster with marginal recall loss on this 50-class problem.
  - Serve behind **NVIDIA Triton Inference Server** with dynamic batching to amortize GPU overhead across concurrent requests.

- **Cross-encoder** (the latency bottleneck):
  - Replace `ms-marco-MiniLM-L-6-v2` (66M params) with a **distilled 2-layer cross-encoder** or a **listwise reranker**; reduces rerank from ~500ms to ~15–25ms on GPU.
  - **Tiered reranking**: skip the cross-encoder when the top RRF candidate's score exceeds a confidence threshold. High-confidence queries (clear specialty match) never pay the rerank cost.
  - For autocomplete specifically: **drop cross-encoder from the hot path** and rerank asynchronously, serving the cached RRF result until the reranked result replaces it.

- **FAISS index**:
  - Replace `IndexFlatIP` with **`IndexHNSWFlat`** or **`IndexIVF_PQ`** for approximate nearest-neighbor search. At 1,500 vectors this is not the bottleneck today, but HNSW is necessary once the corpus scales to millions.
  - Keep the index **memory-mapped and read-only**; replicate per pod at startup. No shared-state required.

### 3. Horizontal scaling

- **Stateless replicas**: each pod loads the FAISS index, BM25 object, and models at startup; all state is read-only in-memory. Zero inter-pod coordination.
- **Load balancer**: Route 5,000 RPS across N pods. At ~8ms/query GPU latency with a batch size of 32, one A10G GPU handles ~4,000 QPS. Two GPU pods + autoscaling covers the target.
- **Corpus updates** (taxonomy changes): rebuild the index offline, push new artifact blob to object storage (S3), trigger rolling pod restart. Zero downtime.

### 4. API design

- **Endpoint**: `GET /classify?q=<query>[&top_k=3]` — read-only, cacheable, GET semantics.
- **Client-side debounce**: fire only after 150–200ms of no keystrokes. Eliminates ~60% of requests at source.
- **Response contract**: `{"results": [{"class": "...", "score": 0.88}, ...], "cached": true/false}` — same top-3 structure as the current CLI output.
- **Timeout + graceful degradation**: 45ms hard timeout; on timeout, fall back to (1) cached result if available, (2) BM25-only result, (3) "General Practice" default. Never fail open with a 500.

### 5. Observability and reliability

- **p99 latency SLO** tracked per stage via OpenTelemetry spans. Alert if query embedding p99 > 15ms or total p99 > 45ms.
- **Cache hit rate** as a primary metric; sudden drops indicate query distribution shift or cache invalidation bug.
- **Canary deploys** for corpus updates: route 5% of traffic to new index, compare top-1 class distribution before full rollout.
- **Model versioning**: artifact bundles (index + BM25 + config) are versioned together. Rollback = swap the artifact pointer; no code deploy required.
