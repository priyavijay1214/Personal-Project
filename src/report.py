from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .config import FIGURES_DIR, REPORTS_DIR

def plot_monthly_total(df: pd.DataFrame) -> Path:
    monthly = df.groupby("month", as_index=False)["amount"].sum().rename(columns={"amount":"total_spend"})
    fig_path = FIGURES_DIR / "monthly_total_spend.png"

    plt.figure()
    plt.plot(monthly["month"], monthly["total_spend"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("Total Spend ($)")
    plt.title("Monthly Total Spend")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path

def plot_category_breakdown(df: pd.DataFrame) -> Path:
    cat = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    fig_path = FIGURES_DIR / "category_breakdown.png"

    plt.figure()
    plt.bar(cat["category"], cat["amount"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Category")
    plt.ylabel("Total Spend ($)")
    plt.title("Spending by Category (All Time)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path

def write_text_report(insights: dict) -> Path:
    out_path = REPORTS_DIR / "summary_report.txt"
    lines = []
    lines.append("Student Spending Analyzer — Summary Report\n")
    for k, v in insights.items():
        lines.append(f"- {k}: {v}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
