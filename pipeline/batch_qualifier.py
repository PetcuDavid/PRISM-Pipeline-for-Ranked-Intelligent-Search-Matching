import pandas as pd
import json
import os
import ollama
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemma3:4b"
BATCH_SIZE = 10


QUALIFIER_PROMPT = """You are a company qualification expert.

Query: "{query}"

Query context:
- Industries: {industries}
- Looking for: {implicit_signals}
- NOT looking for: {negative_terms}

Rate each company below from 0 to 10 on how well it matches the query.
Return ONLY a valid JSON array, no markdown, no explanation.

Companies to evaluate:
{companies}

Return this exact format:
[
  {{"id": 1, "score": 8, "reason": "one sentence explanation"}},
  {{"id": 2, "score": 2, "reason": "one sentence explanation"}}
]"""


def _call_local_llm(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    return response["message"]["content"].strip()


def _serialize_for_llm(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get("operational_name")):
        parts.append(f"Name: {row['operational_name']}")
    if pd.notna(row.get("description")):
        parts.append(f"Description: {str(row['description'])[:300]}")
    if isinstance(row.get("core_offerings"), list):
        parts.append(f"Offerings: {', '.join(row['core_offerings'])}")
    if isinstance(row.get("target_markets"), list):
        parts.append(f"Markets: {', '.join(row['target_markets'])}")
    if isinstance(row.get("business_model"), list):
        parts.append(f"Model: {', '.join(row['business_model'])}")
    naics = row.get("primary_naics")
    if isinstance(naics, dict):
        parts.append(f"Industry: {naics.get('label', '')}")
    return " | ".join(parts)


def _qualify_batch(batch: pd.DataFrame, analysis: dict) -> list[dict]:
    companies_text = ""
    for i, (_, row) in enumerate(batch.iterrows(), 1):
        companies_text += f"[{i}] {_serialize_for_llm(row)}\n"

    prompt = QUALIFIER_PROMPT.format(
        query=analysis["original_query"],
        industries=", ".join(analysis.get("target_industries", [])),
        implicit_signals=", ".join(analysis.get("implicit_signals", [])[:5]),
        negative_terms=", ".join(analysis.get("negative_terms", [])[:5]),
        companies=companies_text
    )

    raw = _call_local_llm(prompt)

    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break
    raw = raw.strip()

    return json.loads(raw)


def qualify_gray_zone(df: pd.DataFrame, analysis: dict) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result_df = df.copy()
    result_df["llm_score"] = 0.0
    result_df["llm_reason"] = ""
    result_df["qualification"] = "AUTO_REJECTED"

    batches = [df.iloc[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    print(f"  Processing {len(df)} companies in {len(batches)} batches...")

    all_results = []
    for batch_num, batch in enumerate(batches, 1):
        print(f"  Batch {batch_num}/{len(batches)} ({len(batch)} companies)...")
        try:
            results = _qualify_batch(batch, analysis)
            all_results.extend(results)
        except Exception as e:
            print(f"  Batch {batch_num} failed: {e}")
            for i in range(len(batch)):
                all_results.append({"id": i+1, "score": 5, "reason": "batch failed"})

    indices = list(df.index)
    for i, res in enumerate(all_results):
        if i >= len(indices):
            break
        idx = indices[i]
        llm_score = float(res.get("score", 0)) / 10.0
        result_df.at[idx, "llm_score"] = llm_score
        result_df.at[idx, "llm_reason"] = res.get("reason", "")

        score_col = "final_score" if "final_score" in result_df.columns else "semantic_score"
        combined = 0.5 * result_df.at[idx, score_col] + 0.5 * llm_score

        if combined >= 0.55:
            result_df.at[idx, "qualification"] = "QUALIFIED"
            result_df.at[idx, "confidence"] = "MEDIUM"
        else:
            result_df.at[idx, "qualification"] = "REJECTED"
            result_df.at[idx, "confidence"] = "MEDIUM"

    return result_df