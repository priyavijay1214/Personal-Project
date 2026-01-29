from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .config import RAW_DIR, CATEGORIES
from .utils import clean_merchant_name

MERCHANTS = {
    "Groceries": ["HEB", "TARGET", "WALMART", "TRADER JOES"],
    "Dining": ["CHIPOTLE", "CAVA", "RAISING CANES", "P TERRYS", "TACODELI"],
    "Coffee": ["STARBUCKS", "DUNKIN", "LOCAL CAFE"],
    "Transport": ["UBER", "LYFT", "CAPMETRO"],
    "Entertainment": ["AMC", "SPOTIFY", "NETFLIX", "BOWLING"],
    "Shopping": ["AMAZON", "NIKE", "URBAN OUTFITTERS"],
    "Health": ["CVS", "WALGREENS", "GYM"],
    "Bills": ["PHONE BILL", "INTERNET", "UTILITIES"],
    "Education": ["TEXTBOOKS", "COURSE MATERIALS"],
    "Misc": ["VENMO", "ZELLE", "OTHER"]
}

def _sample_amount(cat: str, rng: np.random.Generator) -> float:
    # lognormal-ish, different typical sizes per category
    base = {
        "Groceries": (3.2, 0.5),
        "Dining": (3.0, 0.55),
        "Coffee": (2.2, 0.35),
        "Transport": (2.7, 0.6),
        "Entertainment": (2.8, 0.7),
        "Shopping": (3.1, 0.8),
        "Health": (2.6, 0.7),
        "Bills": (3.3, 0.35),
        "Education": (3.4, 0.9),
        "Misc": (2.7, 0.9),
    }[cat]
    amt = float(rng.lognormal(mean=base[0], sigma=base[1]))
    # clamp and round
    return round(min(max(amt, 2.5), 350.0), 2)

def generate_transactions(
    start_date: str = "2025-08-01",
    end_date: str = "2026-01-15",
    seed: int = 7,
    out_name: str = "transactions_sample.csv",
) -> str:
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    days = (end - start).days + 1
    rows = []

    for i in range(days):
        d = start + timedelta(days=i)

        # spending frequency: more on weekends
        weekday = d.weekday()  # 0=Mon
        lam = 1.3 if weekday < 5 else 2.2
        n = int(rng.poisson(lam=lam))

        for _ in range(n):
            cat = rng.choice(CATEGORIES, p=[0.14,0.16,0.10,0.09,0.10,0.12,0.06,0.12,0.05,0.06])
            merchant = rng.choice(MERCHANTS.get(cat, ["OTHER"]))
            amount = _sample_amount(cat, rng)

            rows.append({
                "date": d.date().isoformat(),
                "merchant": clean_merchant_name(merchant),
                "category": cat,
                "amount": amount,
                "payment_method": rng.choice(["Debit", "Credit", "Cash"], p=[0.52, 0.42, 0.06])
            })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out_path = RAW_DIR / out_name
    df.to_csv(out_path, index=False)
    return str(out_path)

if __name__ == "__main__":
    print(generate_transactions())
