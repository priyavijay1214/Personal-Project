import pandas as pd
from .utils import assert_columns, parse_dates, clean_merchant_name

REQUIRED = ["date", "merchant", "category", "amount", "payment_method"]

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    assert_columns(df, REQUIRED)
    df = df.copy()

    df = parse_dates(df, "date")
    df = df.dropna(subset=["date"])
    df["merchant"] = df["merchant"].map(clean_merchant_name)
    df["category"] = df["category"].astype(str).str.strip()
    df["payment_method"] = df["payment_method"].astype(str).str.strip()

    # amount must be positive
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > 0]

    # add time fields
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["dow"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.weekday >= 5

    return df.reset_index(drop=True)
