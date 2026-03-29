import pandas as pd
from typing import Optional

COUNTRY_TO_CODE = {
    "romania": "ro", "germany": "de", "france": "fr", "spain": "es",
    "italy": "it", "poland": "pl", "netherlands": "nl", "sweden": "se",
    "switzerland": "ch", "norway": "no", "denmark": "dk", "finland": "fi",
    "austria": "at", "belgium": "be", "portugal": "pt", "czech": "cz",
    "hungary": "hu", "greece": "gr", "ireland": "ie", "luxembourg": "lu",
    "united kingdom": "gb", "uk": "gb", "united states": "us", "usa": "us",
    "canada": "ca", "mexico": "mx", "china": "cn", "japan": "jp",
    "india": "in", "south korea": "kr", "singapore": "sg", "brazil": "br",
    "australia": "au",
}

CONTINENT_TO_CODES = {
    "europe": ["ro", "de", "fr", "es", "it", "pl", "nl", "se", "ch", "no",
               "dk", "fi", "at", "be", "pt", "cz", "hu", "gr", "ie", "lu", "gb"],
    "north america": ["us", "ca", "mx"],
    "asia": ["cn", "jp", "in", "kr", "sg"],
    "south america": ["br", "ar", "cl", "co"],
    "oceania": ["au", "nz"],
}


def _get_country_code(address) -> Optional[str]:
    if address is None:
        return None
    if isinstance(address, dict):
        return address.get("country_code", None)
    if isinstance(address, str):
        addr_lower = address.lower()
        for country_name, code in COUNTRY_TO_CODE.items():
            if country_name in addr_lower:
                return code
    return None


def _passes_filter(value, condition) -> Optional[bool]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return condition(value)
    except Exception:
        return None


def apply_hard_filters(df: pd.DataFrame, filters: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), pd.DataFrame()

    mask_keep = pd.Series([True] * len(df), index=df.index)
    mask_penalty = pd.Series([False] * len(df), index=df.index)

    country_codes = df["address"].apply(_get_country_code)

    country = filters.get("country")
    if country:
        target_code = COUNTRY_TO_CODE.get(country.lower())
        if target_code:
            results = country_codes.apply(
                lambda code: None if code is None else (code == target_code)
            )
            mask_keep &= results != False
            mask_penalty |= results.isna()

    continent = filters.get("continent")
    if continent:
        valid_codes = CONTINENT_TO_CODES.get(continent.lower(), [])
        if valid_codes:
            results = country_codes.apply(
                lambda code: None if code is None else (code in valid_codes)
            )
            mask_keep &= results != False
            mask_penalty |= results.isna()

    min_employees = filters.get("min_employees")
    if min_employees is not None:
        results = df["employee_count"].apply(
            lambda x: _passes_filter(x, lambda v: int(v) >= int(min_employees))
        )
        mask_keep &= results != False
        mask_penalty |= results.isna()

    max_employees = filters.get("max_employees")
    if max_employees is not None:
        results = df["employee_count"].apply(
            lambda x: _passes_filter(x, lambda v: int(v) <= int(max_employees))
        )
        mask_keep &= results != False
        mask_penalty |= results.isna()

    min_revenue = filters.get("min_revenue")
    if min_revenue is not None:
        results = df["revenue"].apply(
            lambda x: _passes_filter(x, lambda v: float(v) >= float(min_revenue))
        )
        mask_keep &= results != False
        mask_penalty |= results.isna()

    max_revenue = filters.get("max_revenue")
    if max_revenue is not None:
        results = df["revenue"].apply(
            lambda x: _passes_filter(x, lambda v: float(v) <= float(max_revenue))
        )
        mask_keep &= results != False
        mask_penalty |= results.isna()

    is_public = filters.get("is_public")
    if is_public is not None:
        results = df["is_public"].apply(
            lambda x: _passes_filter(x, lambda v: bool(v) == bool(is_public))
        )
        mask_keep &= results != False
        mask_penalty |= results.isna()

    founded_after = filters.get("founded_after")
    if founded_after is not None:
        results = df["year_founded"].apply(
            lambda x: _passes_filter(x, lambda v: int(v) >= int(founded_after))
        )
        mask_keep &= results != False
        mask_penalty |= results.isna()

    founded_before = filters.get("founded_before")
    if founded_before is not None:
        results = df["year_founded"].apply(
            lambda x: _passes_filter(x, lambda v: int(v) <= int(founded_before))
        )
        mask_keep &= results != False
        mask_penalty |= results.isna()

    passed_df = df[mask_keep].copy()
    passed_df["has_missing_fields"] = mask_penalty[mask_keep]
    rejected_df = df[~mask_keep].copy()

    return passed_df, rejected_df


def print_filter_stats(original: pd.DataFrame, passed: pd.DataFrame, rejected: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"HARD FILTER RESULTS")
    print(f"{'='*60}")
    print(f"Original:  {len(original)} companies")
    print(f"Passed:    {len(passed)} companies")
    print(f"Rejected:  {len(rejected)} companies")
    if len(passed) > 0:
        penalized = passed["has_missing_fields"].sum()
        print(f"Uncertain: {penalized} (missing fields, penalized in scoring)")


if __name__ == "__main__":
    df = pd.read_json("data/companies.jsonl", lines=True)

    test_cases = [
        {
            "query": "Logistic companies in Romania",
            "filters": {"country": "Romania"}
        },
        {
            "query": "Public software companies with more than 1,000 employees",
            "filters": {"is_public": True, "min_employees": 1001}
        },
        {
            "query": "Construction companies in US with revenue over $50 million",
            "filters": {"country": "United States", "min_revenue": 50000000}
        }
    ]

    for test in test_cases:
        print(f"\nQuery: {test['query']}")
        passed, rejected = apply_hard_filters(df, test["filters"])
        print_filter_stats(df, passed, rejected)
        if len(passed) > 0:
            print("Primele 3 companii care au trecut filtrul:")
            for _, row in passed.head(3).iterrows():
                name = row.get("operational_name", "N/A")
                addr = row.get("address", {})
                town = addr.get("town", "N/A") if isinstance(addr, dict) else addr
                country_code = _get_country_code(addr)
                print(f"  - {name} | {town} | {country_code}")