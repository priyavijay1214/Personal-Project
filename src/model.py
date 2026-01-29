from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

def train_overspend_model(model_df: pd.DataFrame):
    df = model_df.copy()

    y = df["is_overspend"].astype(int)
    X = df.drop(columns=["month", "is_overspend"])

    # tiny dataset (months) → keep it simple
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(1, int(0.3 * len(df))), random_state=42, stratify=None
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "classification_report": classification_report(y_test, preds, output_dict=False),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "test_months": len(X_test),
        "overspend_probabilities": proba.tolist(),
    }

    # feature importance via coefficients
    coef = pd.Series(clf.coef_[0], index=X.columns).sort_values(key=lambda s: s.abs(), ascending=False)
    return clf, metrics, coef
