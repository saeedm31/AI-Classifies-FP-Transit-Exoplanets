"""Evaluate the classifier with cross-validation and save the results.

Usage:
    python train.py

Writes metrics.json, which the app's "Model performance" tab displays.
The app itself trains the final model on all clean data at startup,
using the same settings (see exoclf.build_model).
"""

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import exoclf


def main():
    df, data_stats = exoclf.load_clean_data()
    X = df[exoclf.MODEL_FEATURES]
    y = (df[exoclf.LABEL] == exoclf.POSITIVE).astype(int).to_numpy()

    print(f"Clean data: {data_stats['clean_rows']} objects "
          f"(removed {data_stats['removed_duplicates']} duplicates, "
          f"{data_stats['removed_conflicting']} conflicting rows)")
    print(f"Classes: {data_stats['class_counts']}")

    # Out-of-fold probabilities: every object is scored by a model that never saw it.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=exoclf.RANDOM_STATE)
    proba = cross_val_predict(exoclf.build_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    baseline = max(y.mean(), 1 - y.mean())
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "precision_fp": precision_score(y, pred),
        "recall_fp": recall_score(y, pred),
        "recall_confirmed": recall_score(y, pred, pos_label=0),
        "f1_fp": f1_score(y, pred),
        "baseline_accuracy": baseline,
    }
    for k, v in metrics.items():
        print(f"{k:20s} {v:.3f}")

    fpr, tpr, _ = roc_curve(y, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))  # thin the curve for the file

    final_model = exoclf.train(df)
    importance = dict(zip(exoclf.MODEL_FEATURES, final_model.feature_importances_.round(4).tolist()))

    out = {
        "data": data_stats,
        "validation": "5-fold stratified cross-validation (out-of-fold predictions, threshold 0.5)",
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        # rows = actual [Confirmed, FP], columns = predicted [Confirmed, FP]
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "roc_curve": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        "feature_importance": importance,
    }
    with open("metrics.json", "w") as f:
        # one line per top-level key keeps the file short and easy to diff
        f.write("{\n" + ",\n".join(f'  "{k}": {json.dumps(v)}' for k, v in out.items()) + "\n}\n")
    print("Saved metrics.json")


if __name__ == "__main__":
    main()
