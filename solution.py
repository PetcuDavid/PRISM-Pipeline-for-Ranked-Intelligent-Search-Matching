import pandas as pd
import json
import os
import sys
import time
from tqdm import tqdm

sys.path.append(".")
from pipeline.query_analyst import analyze_query
from pipeline.hard_filter import apply_hard_filters
from pipeline.semantic_ranker import rank_companies
from pipeline.cross_encoder_reranker import rerank
from pipeline.confidence_splitter import split_by_confidence
from pipeline.batch_qualifier import qualify_gray_zone

QUERIES = [
    "Logistic companies in Romania",
    "Public software companies with more than 1,000 employees",
    "Food and beverage manufacturers in France",
    "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    "Construction companies in the United States with revenue over $50 million",
    "Pharmaceutical companies in Switzerland",
    "B2B SaaS companies providing HR solutions in Europe",
    "Clean energy startups founded after 2018 with fewer than 200 employees",
    "Fast-growing fintech companies competing with traditional banks in Europe",
    "E-commerce companies using Shopify or similar platforms",
    "Renewable energy equipment manufacturers in Scandinavia",
    "Companies that manufacture or supply critical components for electric vehicle battery production"
]

MIN_COMPANIES_AFTER_FILTER = 15


def run_pipeline(df: pd.DataFrame, query: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    print("\n[1] Analyzing query...")
    analysis = analyze_query(query)
    print(f"    Type: {analysis['query_type']} | Difficulty: {analysis['difficulty']}")

    print("\n[2] Applying hard filters...")
    passed, rejected = apply_hard_filters(df, analysis["structured_filters"])
    print(f"    {len(df)} → {len(passed)} companies passed")

    if len(passed) < MIN_COMPANIES_AFTER_FILTER:
        print(f"    Too few companies after filter ({len(passed)} < {MIN_COMPANIES_AFTER_FILTER})")
        print(f"    Fallback: using full dataset for semantic ranking")
        passed = df.copy()
        passed["has_missing_fields"] = False

    print("\n[3] Semantic ranking...")
    ranked = rank_companies(passed, analysis)

    print("\n[4] Cross-encoder reranking...")
    reranked = rerank(ranked, analysis)

    print("\n[5] Confidence splitting...")
    auto_qualified, gray_zone, auto_rejected = split_by_confidence(
        reranked, analysis["difficulty"]
    )

    if not gray_zone.empty:
        print("\n[6] LLM batch qualification for gray zone...")
        gray_qualified = qualify_gray_zone(gray_zone, analysis)
    else:
        print("\n[6] No gray zone companies, skipping LLM...")
        gray_qualified = gray_zone

    final_qualified = pd.concat([
        auto_qualified,
        gray_qualified[gray_qualified["qualification"].isin(["QUALIFIED", "AUTO_QUALIFIED"])]
    ], ignore_index=True)

    if final_qualified.empty:
        return pd.DataFrame()

    score_col = "final_score" if "final_score" in final_qualified.columns else "semantic_score"
    final_qualified = final_qualified.sort_values(score_col, ascending=False).reset_index(drop=True)
    final_qualified["final_rank"] = final_qualified.index + 1

    return final_qualified


def save_results(results: pd.DataFrame, query: str, query_idx: int):
    os.makedirs("results", exist_ok=True)
    safe_query = query[:50].replace(" ", "_").replace("/", "_")
    filename = f"results/query_{query_idx:02d}_{safe_query}.json"

    output = []
    score_col = "final_score" if "final_score" in results.columns else "semantic_score"

    for _, row in results.iterrows():
        name = row.get("operational_name", "N/A")
        if pd.isna(name):
            name = "N/A"

        llm_reason = row.get("llm_reason", "")
        if pd.isna(llm_reason):
            llm_reason = ""

        entry = {
            "rank": int(row["final_rank"]),
            "company": str(name),
            "website": str(row.get("website", "N/A")),
            "score": round(float(row[score_col]), 4),
            "qualification": str(row.get("qualification", "QUALIFIED")),
            "confidence": str(row.get("confidence", "HIGH")),
            "llm_reason": str(llm_reason),
            "description_preview": str(row.get("description", ""))[:150],
        }
        output.append(entry)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"    Saved {len(output)} companies to {filename}")


def print_top_results(results: pd.DataFrame, n: int = 5):
    score_col = "final_score" if "final_score" in results.columns else "semantic_score"
    print(f"\n  Top {min(n, len(results))} results:")
    for _, row in results.head(n).iterrows():
        name = row.get("operational_name", "N/A")
        score = row[score_col]
        qual = row.get("qualification", "")
        reason = row.get("llm_reason", "")
        if pd.isna(reason):
            reason = ""
        print(f"  [{score:.3f}] {name} | {qual}")
        if reason:
            print(f"           {reason}")


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_json("data/companies.jsonl", lines=True)
    print(f"Loaded {len(df)} companies")

    all_results = {}

    for idx, query in enumerate(QUERIES, 1):
        try:
            results = run_pipeline(df, query)
            if not results.empty:
                print_top_results(results)
                save_results(results, query, idx)
                all_results[query] = len(results)
            else:
                all_results[query] = 0
        except Exception as e:
            print(f"  Pipeline failed: {e}")
            all_results[query] = -1

        time.sleep(2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for query, count in all_results.items():
        if count == -1:
            status = "FAILED"
        elif count == 0:
            status = "0 companies qualified"
        else:
            status = f"{count} companies qualified"
        print(f"  {query[:55]}: {status}")