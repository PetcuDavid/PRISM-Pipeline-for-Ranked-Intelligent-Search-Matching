import json
import os
import ollama
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemma3:4b"
CACHE_FILE = "results/query_analysis_cache.json"

ANALYST_PROMPT = """You are an expert business analyst.
Analyze this query about companies: {query}

Return ONLY a valid JSON object, no markdown, no backticks, no explanation.
The JSON must have this exact structure:
{{
    "query_type": "structured or semantic or hybrid",
    "structured_filters": {{
        "country": null,
        "continent": null,
        "min_employees": null,
        "max_employees": null,
        "min_revenue": null,
        "max_revenue": null,
        "is_public": null,
        "founded_after": null,
        "founded_before": null,
        "business_model": null
    }},
    "target_industries": [],
    "naics_codes": [],
    "expanded_terms": [],
    "negative_terms": [],
    "implicit_signals": [],
    "difficulty": "easy or medium or hard",
    "reasoning": ""
}}

Rules:
- structured_filters: integers for numbers, null if not mentioned.
- negative_terms: what a NON-matching company would look like.
- Return raw JSON only. Nothing else."""


def _load_cache() -> dict:
    os.makedirs("results", exist_ok=True)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    os.makedirs("results", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _call_local_llm(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    return response["message"]["content"].strip()


def analyze_query(query: str) -> dict:
    cache = _load_cache()
    if query in cache:
        print(f"    (using cached analysis)")
        return cache[query]

    prompt = ANALYST_PROMPT.format(query=query)
    raw = _call_local_llm(prompt)

    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    raw = raw.strip()

    if not raw:
        raise ValueError(f"Empty response for query: {query}")

    result = json.loads(raw)
    result["original_query"] = query

    cache[query] = result
    _save_cache(cache)

    return result


def print_analysis(analysis: dict):
    print("\n" + "="*60)
    print("QUERY: " + analysis["original_query"])
    print("="*60)
    print("Type:       " + analysis["query_type"])
    print("Difficulty: " + analysis["difficulty"])
    print("Reasoning:  " + analysis["reasoning"])
    print("\nStructured filters:")
    for key, value in analysis["structured_filters"].items():
        if value is not None:
            print("  " + key + ": " + str(value))
    print("Industries:       " + str(analysis["target_industries"]))
    print("Expanded terms:   " + str(analysis["expanded_terms"]))
    print("Negative terms:   " + str(analysis["negative_terms"]))
    print("Implicit signals: " + str(analysis["implicit_signals"]))


if __name__ == "__main__":
    test_queries = [
        "Logistic companies in Romania",
        "Public software companies with more than 1,000 employees",
        "Fast-growing fintech companies competing with traditional banks in Europe"
    ]
    for query in test_queries:
        print(f"\nAnalyzing: {query}")
        analysis = analyze_query(query)
        print_analysis(analysis)