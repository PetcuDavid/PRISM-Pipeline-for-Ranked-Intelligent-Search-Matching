import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def _serialize_company(row: pd.Series) -> str:
    parts = []

    if pd.notna(row.get("operational_name")):
        parts.append(str(row["operational_name"]))

    if pd.notna(row.get("description")):
        parts.append(str(row["description"]))

    if isinstance(row.get("core_offerings"), list):
        parts.append(" ".join(row["core_offerings"]))
    elif pd.notna(row.get("core_offerings")):
        parts.append(str(row["core_offerings"]))

    if isinstance(row.get("target_markets"), list):
        parts.append(" ".join(row["target_markets"]))
    elif pd.notna(row.get("target_markets")):
        parts.append(str(row["target_markets"]))

    if isinstance(row.get("business_model"), list):
        parts.append(" ".join(row["business_model"]))
    elif pd.notna(row.get("business_model")):
        parts.append(str(row["business_model"]))

    naics = row.get("primary_naics")
    if isinstance(naics, dict) and "label" in naics:
        parts.append(naics["label"])

    secondary = row.get("secondary_naics")
    if isinstance(secondary, list):
        for item in secondary:
            if isinstance(item, dict) and "label" in item:
                parts.append(item["label"])

    return " ".join(parts).lower()


def _build_enriched_query(analysis: dict) -> str:
    parts = [analysis["original_query"]]

    if analysis.get("expanded_terms"):
        parts.extend(analysis["expanded_terms"])

    if analysis.get("target_industries"):
        parts.extend(analysis["target_industries"])

    if analysis.get("implicit_signals"):
        parts.extend(analysis["implicit_signals"])

    if analysis.get("naics_codes"):
        parts.extend(analysis["naics_codes"])

    return " ".join(parts).lower()


def compute_bm25_scores(corpus: list[str], query: str) -> np.ndarray:
    tokenized_corpus = [doc.split() for doc in corpus]
    tokenized_query = query.split()
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    return scores


def compute_embedding_scores(corpus: list[str], query: str) -> np.ndarray:
    query_embedding = EMBEDDING_MODEL.encode([query], normalize_embeddings=True)
    corpus_embeddings = EMBEDDING_MODEL.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    scores = (corpus_embeddings @ query_embedding.T).flatten()
    return scores


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def rank_companies(df: pd.DataFrame, analysis: dict, bm25_weight: float = 0.35, embedding_weight: float = 0.65) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    corpus = [_serialize_company(row) for _, row in df.iterrows()]
    enriched_query = _build_enriched_query(analysis)

    print(f"  Computing BM25 scores...")
    bm25_scores = compute_bm25_scores(corpus, enriched_query)

    print(f"  Computing embedding scores...")
    embedding_scores = compute_embedding_scores(corpus, enriched_query)

    bm25_norm = normalize_scores(bm25_scores)
    embedding_norm = normalize_scores(embedding_scores)

    combined = (bm25_weight * bm25_norm) + (embedding_weight * embedding_norm)

    result = df.copy()
    result["bm25_score"] = bm25_norm
    result["embedding_score"] = embedding_norm
    result["semantic_score"] = combined

    if "has_missing_fields" in result.columns:
        result.loc[result["has_missing_fields"] == True, "semantic_score"] *= 0.85

    result = result.sort_values("semantic_score", ascending=False).reset_index(drop=True)
    result["semantic_rank"] = result.index + 1

    return result


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from pipeline.query_analyst import analyze_query
    from pipeline.hard_filter import apply_hard_filters

    df = pd.read_json("data/companies.jsonl", lines=True)

    test_queries = [
        "Logistic companies in Romania",
        "Public software companies with more than 1,000 employees",
        "Fast-growing fintech companies competing with traditional banks in Europe"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        analysis = analyze_query(query)
        passed, rejected = apply_hard_filters(df, analysis["structured_filters"])

        print(f"After hard filter: {len(passed)} companies")
        ranked = rank_companies(passed, analysis)

        print(f"\nTop 5 rezultate:")
        for _, row in ranked.head(5).iterrows():
            name = row.get("operational_name", "N/A")
            score = row["semantic_score"]
            desc = str(row.get("description", ""))[:80]
            print(f"  [{score:.3f}] {name}")
            print(f"           {desc}...")