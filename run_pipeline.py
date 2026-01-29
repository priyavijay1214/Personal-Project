import pandas as pd
from src.generate_sample_data import generate_transactions
from src.clean import clean_transactions
from src.features import monthly_summary, build_model_table
from src.model import train_overspend_model
from src.report import plot_monthly_total, plot_category_breakdown, write_text_report
from src.config import RAW_DIR, PROCESSED_DIR

def main():
    # 1) Generate sample data (or replace with your real CSV later)
    csv_path = generate_transactions()
    print(f"Generated sample data: {csv_path}")

    # 2) Load + clean
    raw = pd.read_csv(csv_path)
    df = clean_transactions(raw)

    # 3) Save processed
    processed_path = PROCESSED_DIR / "transactions_clean.csv"
    df.to_csv(processed_path, index=False)
    print(f"Saved cleaned data: {processed_path}")

    # 4) Summaries
    ms = monthly_summary(df)
    ms_path = PROCESSED_DIR / "monthly_summary.csv"
    ms.to_csv(ms_path, index=False)
    print(f"Saved monthly summary: {ms_path}")

    # 5) Model
    model_df = build_model_table(df, overspend_threshold=1200.0)
    clf, metrics, coef = train_overspend_model(model_df)

    # 6) Plots + report
    p1 = plot_monthly_total(df)
    p2 = plot_category_breakdown(df)

    top_drivers = ", ".join([f"{idx} ({coef[idx]:.2f})" for idx in coef.index[:5]])
    insights = {
        "Rows (transactions)": len(df),
        "Months": df["month"].nunique(),
        "Total spend ($)": round(df["amount"].sum(), 2),
        "Avg transaction ($)": round(df["amount"].mean(), 2),
        "Model test months": metrics["test_months"],
        "Confusion matrix": metrics["confusion_matrix"],
        "Top overspend drivers (coef)": top_drivers,
        "Figures": f"{p1.name}, {p2.name}",
    }
    rpt = write_text_report(insights)
    print(f"Report written: {rpt}")

    print("\n=== Classification Report ===")
    print(metrics["classification_report"])

if __name__ == "__main__":
    main()
