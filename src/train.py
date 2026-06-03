"""Train the block-field classifier and log to MLflow.

Model: a ColumnTransformer (TF-IDF on text + char n-grams on class tokens +
scaled numeric features) into LogisticRegression. It's small, CPU-only, trains
in <1s, and exposes calibrated-ish probabilities we use as confidence at serving
time. Every run is tracked in MLflow and the fitted pipeline is registered, so
deploys and rollbacks are reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config, load_config
from .dataset import build_dataset
from .features import CLASS_FEATURE, NUMERIC_FEATURES, TEXT_FEATURE


def build_pipeline() -> Pipeline:
    """The estimator. Kept in one place so train + serve agree exactly."""
    pre = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                     max_features=5000, sublinear_tf=True),
             TEXT_FEATURE),
            ("cls", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                    min_df=1),
             CLASS_FEATURE),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("features", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   C=4.0)),
    ])


def _load_frame(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return pd.DataFrame(rows)


def train(cfg: Config, rebuild_data: bool = True) -> dict:
    if rebuild_data:
        data_path = build_dataset(cfg)
    else:
        data_path = build_dataset(cfg, merge_feedback=True)

    df = _load_frame(data_path)
    X, y = df.drop(columns=["label"]), df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(cfg.mlops.mlflow_tracking_uri)
    mlflow.set_experiment(f"scraper-{cfg.target.name}")

    with mlflow.start_run() as run:
        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)

        preds = pipe.predict(X_te)
        macro_f1 = f1_score(y_te, preds, average="macro")
        report = classification_report(y_te, preds, output_dict=True,
                                       zero_division=0)

        mlflow.log_params({
            "model": "logreg",
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "fields": ",".join(cfg.schema_.field_names),
        })
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("accuracy", report["accuracy"])
        for label in cfg.schema_.labels:
            if label in report:
                mlflow.log_metric(f"f1_{label}", report[label]["f1-score"])

        # Persist a feature baseline for drift monitoring later.
        baseline = {c: {"mean": float(X_tr[c].mean()),
                        "std": float(X_tr[c].std() or 1.0)}
                    for c in NUMERIC_FEATURES}
        mlflow.log_dict(baseline, "feature_baseline.json")

        mlflow.sklearn.log_model(
            pipe, "model",
            registered_model_name=cfg.mlops.registered_model_name,
        )
        print(f"[train] macro_f1={macro_f1:.3f}  run_id={run.info.run_id}")
        return {"run_id": run.info.run_id, "macro_f1": macro_f1,
                "baseline": baseline}


if __name__ == "__main__":
    train(load_config())
