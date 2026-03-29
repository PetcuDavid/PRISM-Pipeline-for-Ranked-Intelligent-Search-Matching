import pandas as pd
import numpy as np


THRESHOLDS = {
    "easy":   {"auto_qualify": 0.75, "auto_reject": 0.30},
    "medium": {"auto_qualify": 0.70, "auto_reject": 0.25},
    "hard":   {"auto_qualify": 0.60, "auto_reject": 0.20},
}


def split_by_confidence(df: pd.DataFrame, difficulty: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = THRESHOLDS.get(difficulty, THRESHOLDS["medium"])
    high = thresholds["auto_qualify"]
    low = thresholds["auto_reject"]

    score_col = "final_score" if "final_score" in df.columns else "semantic_score"

    auto_qualify = df[df[score_col] >= high].copy()
    auto_reject = df[df[score_col] < low].copy()
    gray_zone = df[(df[score_col] >= low) & (df[score_col] < high)].copy()

    auto_qualify["qualification"] = "AUTO_QUALIFIED"
    auto_qualify["confidence"] = "HIGH"
    auto_reject["qualification"] = "AUTO_REJECTED"
    auto_reject["confidence"] = "HIGH"
    gray_zone["qualification"] = "PENDING_LLM"
    gray_zone["confidence"] = "LOW"

    print(f"\n  Difficulty: {difficulty}")
    print(f"  Thresholds: qualify>={high} | reject<{low}")
    print(f"  Auto-qualified: {len(auto_qualify)}")
    print(f"  Gray zone:      {len(gray_zone)} → going to LLM")
    print(f"  Auto-rejected:  {len(auto_reject)}")

    return auto_qualify, gray_zone, auto_reject