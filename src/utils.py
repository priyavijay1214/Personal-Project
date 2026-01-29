import re
import pandas as pd

def clean_merchant_name(name: str) -> str:
    if not isinstance(name, str):
        return "UNKNOWN"
    name = name.strip().upper()
    name = re.sub(r"[^A-Z0-9 &'-]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name

def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
