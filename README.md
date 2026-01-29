# Personal-Project

# Student Spending Analyzer (Python + Pandas)

A data analytics project that cleans transaction data, produces monthly/category insights, and predicts whether a month is likely to be an **overspending month** using a simple Logistic Regression model.

## What it does
- Generates or ingests transaction-level spending data
- Cleans + validates data (dates, amounts, categories)
- Produces monthly trends + category breakdowns
- Builds month-level features (category spend, transaction behavior)
- Trains a baseline model to flag overspending months

## Tech stack
Python, pandas, matplotlib, scikit-learn

## Quickstart
```bash
pip install -r requirements.txt
python run_pipeline.py
