import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K_RERANK = 30


def _build_rerank_query(analysis: dict) -> str:
    parts = [analysis["original_query"]]
    if analysis.get("expanded_terms"):
        parts.append(" ".join(analysis["expanded_terms"][:5]))
    if analysis.get("implicit_signals"):
        parts.append(" ".join(analysis["implicit_signals"][:3]))
    return " ".join(parts)


def _serialize_for_reranker(row: pd.Series) -> str:
    parts = []

    if pd.notna(row.get("operational_name")):
        parts.append(f"Company: {row['operational_name']}")

    if pd.notna(row.get("description")):
        desc = str(row["description"])[:400]
        parts.append(f"Description: {desc}")

    if isinstance(row.get("core_offerings"), list):
        parts.append(f"Offerings: {', '.join(row['core_offerings'])}")

    if isinstance(row.get("target_markets"), list):
        parts.append(f"Markets: {', '.join(row['target_markets'])}")

    naics = row.get("primary_naics")
    if isinstance(naics, dict) and "label" in naics:
        parts.append(f"Industry: {naics['label']}")

    return " | ".join(parts)


def rerank(df: pd.DataFrame, analysis: dict, top_k: int = TOP_K_RERANK) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    candidates = df.head(top_k).copy()
    rest = df.iloc[top_k:].copy()

    query = _build_rerank_query(analysis)
    passages = [_serialize_for_reranker(row) for _, row in candidates.iterrows()]

    pairs = [[query, passage] for passage in passages]

    print(f"  Cross-encoding {len(pairs)} pairs...")
    scores = CROSS_ENCODER_MODEL.predict(pairs, show_progress_bar=False)

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    candidates["cross_encoder_score"] = scores_norm
    candidates["final_score"] = (
        0.4 * candidates["semantic_score"] +
        0.6 * candidates["cross_encoder_score"]
    )

    if len(rest) > 0:
        rest["cross_encoder_score"] = 0.0
        rest["final_score"] = 0.4 * rest["semantic_score"]

    result = pd.concat([candidates, rest], ignore_index=True)
    result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
    result["final_rank"] = result.index + 1

    return result


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from pipeline.query_analyst import analyze_query
    from pipeline.hard_filter import apply_hard_filters
    from pipeline.semantic_ranker import rank_companies

    df = pd.read_json("data/companies.jsonl", lines=True)

    test_queries = [
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
        print(f"After semantic ranking: top {min(30, len(ranked))} go to cross-encoder")

        reranked = rerank(ranked, analysis)

        print(f"\nTop 5 dupa cross-encoder reranking:")
        for _, row in reranked.head(5).iterrows():
            name = row.get("operational_name", "N/A")
            final = row["final_score"]
            semantic = row["semantic_score"]
            cross = row["cross_encoder_score"]
            desc = str(row.get("description", ""))[:80]
            print(f"  [{final:.3f}] {name}")
            print(f"           semantic={semantic:.3f} | cross={cross:.3f}")
            print(f"           {desc}...")