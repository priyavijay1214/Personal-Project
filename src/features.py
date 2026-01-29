import pandas as pd

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df.groupby(["month", "category"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "monthly_spend"})
    )
    total = (
        df.groupby("month", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "total_spend"})
    )
    out = m.merge(total, on="month", how="left")
    out["share_of_month"] = out["monthly_spend"] / out["total_spend"]
    return out.sort_values(["month", "monthly_spend"], ascending=[True, False])

def build_model_table(df: pd.DataFrame, overspend_threshold: float = 1200.0) -> pd.DataFrame:
    # one row per month, with category spends as features
    pivot = (
        df.groupby(["month", "category"])["amount"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    pivot["total_spend"] = pivot.drop(columns=["month"]).sum(axis=1)
    pivot["is_overspend"] = (pivot["total_spend"] >= overspend_threshold).astype(int)

    # add behavior features
    tx = df.groupby("month", as_index=False).agg(
        n_transactions=("amount", "size"),
        avg_transaction=("amount", "mean"),
        weekend_share=("is_weekend", "mean"),
    )
    out = pivot.merge(tx, on="month", how="left")
    return out
