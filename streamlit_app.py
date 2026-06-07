from importlib import import_module
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CMS Query Understanding Pipeline",
    page_icon="🩺",
    layout="wide",
)


def load_module(module_name: str):
    try:
        return import_module(module_name)
    except Exception as exc:
        st.error(f"Failed to import `{module_name}`: {exc}")
        return None


@st.cache_resource
def load_search_artifacts():
    search_module = import_module("search")
    return search_module.load_artifacts()


def render_search_tab() -> None:
    st.subheader("Search")
    st.write("Run the retrieval and reranking pipeline against the indexed CMS specialty artifacts.")

    query = st.text_input("Patient query", placeholder="I need a doctor for my bad back")
    run_search = st.button("Run Search", type="primary")

    if not run_search:
        return

    if not query.strip():
        st.error("Enter a non-empty query.")
        return

    search_module = load_module("search")
    if search_module is None:
        return

    try:
        result = search_module.predict_query(
            query=query,
            artifacts=load_search_artifacts(),
            score_top_n=3,
        )
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    top3_df = pd.DataFrame(
        result["top3"],
        columns=["CMS Specialty", "Confidence Score"],
    )

    st.success(result["formatted_output"])
    st.dataframe(top3_df, use_container_width=True, hide_index=True)

    with st.expander("Verbose pipeline details"):
        st.write("Scores shown are independent relevance/confidence scores, not softmax probabilities.")

        st.write("Top FAISS hits")
        faiss_df = pd.DataFrame(
            [
                {
                    "score": round(score, 4),
                    "class": result["metadata"][idx]["class"],
                    "matched_query": result["metadata"][idx]["query"],
                }
                for idx, score in result["dense_hits"][:10]
            ]
        )
        st.dataframe(faiss_df, use_container_width=True, hide_index=True)

        st.write("Top BM25 hits")
        bm25_df = pd.DataFrame(
            [
                {
                    "score": round(score, 4),
                    "class": result["metadata"][idx]["class"],
                    "matched_query": result["metadata"][idx]["query"],
                }
                for idx, score in result["sparse_hits"][:10]
            ]
        )
        st.dataframe(bm25_df, use_container_width=True, hide_index=True)

        st.write("Top RRF candidates")
        rrf_df = pd.DataFrame(
            [
                {
                    "rrf_score": round(score, 5),
                    "class": result["metadata"][idx]["class"],
                    "matched_query": result["metadata"][idx]["query"],
                }
                for idx, score in result["rrf_candidates"][:20]
            ]
        )
        st.dataframe(rrf_df, use_container_width=True, hide_index=True)

        st.write("Top cross-encoder reranked candidates")

        reranked_rows = []
        for item in result["reranked"][:20]:
            if isinstance(item, dict):
                idx = item["idx"]
                reranked_rows.append(
                    {
                        "ce_score": round(item.get("ce_score", 0.0), 4),
                        "rrf_score": round(item.get("rrf_score", 0.0), 5),
                        "rrf_norm": round(item.get("rrf_norm", 0.0), 4),
                        "final_score": round(item.get("final_score", item.get("ce_score", 0.0)), 4),
                        "class": result["metadata"][idx]["class"],
                        "matched_query": result["metadata"][idx]["query"],
                    }
                )
            else:
                idx, score = item
                reranked_rows.append(
                    {
                        "ce_score": round(score, 4),
                        "class": result["metadata"][idx]["class"],
                        "matched_query": result["metadata"][idx]["query"],
                    }
                )

        reranked_df = pd.DataFrame(reranked_rows)
        st.dataframe(reranked_df, use_container_width=True, hide_index=True)

        st.write("Final class scores")
        score_df = pd.DataFrame(
            [
                {
                    "class": cls,
                    "confidence_score": round(score, 4),
                }
                for cls, score in sorted(
                    result["class_scores"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            ]
        )
        st.dataframe(score_df, use_container_width=True, hide_index=True)

def render_generate_tab() -> None:
    st.subheader("Generate Synthetic Queries")
    st.write("Create `synthetic_queries.csv` from the CMS taxonomy using the configured Hugging Face model.")

    generate_module = load_module("generate_data")
    if generate_module is None:
        return

    left, right = st.columns([2, 1])
    with left:
        model_name = st.text_input("Generation model", value=generate_module.MODEL_NAME)
        output_file = st.text_input("Output CSV", value=str(generate_module.OUTPUT_FILE))
    with right:
        queries_per_specialty = st.number_input(
            "Queries per specialty",
            min_value=5,
            max_value=100,
            value=generate_module.QUERIES_PER_SPECIALTY,
            step=5,
        )

    hf_token = st.text_input("HF token", value=generate_module.HF_TOKEN or "", type="password")

    try:
        specialties = generate_module.load_top_50()
    except Exception as exc:
        st.error(f"Failed to load top specialties: {exc}")
        return

    st.caption(f"Top specialties loaded: {len(specialties)}")
    st.dataframe(pd.DataFrame({"specialty": specialties}), use_container_width=True, height=220, hide_index=True)

    if not st.button("Generate Dataset"):
        return

    progress_bar = st.progress(0.0)
    status = st.empty()

    def progress_callback(stage: str, current: int, total: int, message: str) -> None:
        if total > 0:
            progress_bar.progress(min(current / total, 1.0))
        status.info(f"[{stage}] {message}")

    try:
        df = generate_module.generate_dataset(
            queries_per_specialty=int(queries_per_specialty),
            output_file=Path(output_file),
            specialties=specialties,
            model_name=model_name,
            hf_token=hf_token or None,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        st.error(f"Dataset generation failed: {exc}")
        return

    progress_bar.progress(1.0)
    status.success(f"Saved {len(df)} rows to {output_file}")
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)


def render_train_tab() -> None:
    st.subheader("Train And Index")
    st.write("Build the FAISS, BM25, metadata, label encoder, and config artifacts from the synthetic dataset.")

    train_module = load_module("train_and_index")
    if train_module is None:
        return

    csv_path = st.text_input("Synthetic dataset path", value=str(train_module.CSV_PATH))

    if not st.button("Build Artifacts"):
        return

    status = st.empty()

    def progress_callback(stage: str, message: str) -> None:
        status.info(f"[{stage}] {message}")

    try:
        config = train_module.run_training_pipeline(csv_path=Path(csv_path), progress_callback=progress_callback)
        load_search_artifacts.clear()
    except Exception as exc:
        st.error(f"Training/indexing failed: {exc}")
        return

    status.success("Artifacts saved successfully.")
    summary = {
        "embedding_model": config["embedding_model"],
        "cross_encoder_model": config["cross_encoder_model"],
        "num_samples": config["num_samples"],
        "num_classes": config["num_classes"],
        "top_k_retrieval": config["top_k_retrieval"],
        "rerank_top_n": config["rerank_top_n"],
    }
    st.json(summary)


def main() -> None:
    st.title("CMS Query Understanding")
    st.caption("Streamlit interface for synthetic data generation, index building, and CMS specialty search.")

    search_tab, generate_tab, train_tab = st.tabs(
        ["Search", "Generate Data", "Train & Index"]
    )

    with search_tab:
        render_search_tab()
    with generate_tab:
        render_generate_tab()
    with train_tab:
        render_train_tab()


if __name__ == "__main__":
    main()
