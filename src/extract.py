"""Inference layer: page URL -> list of structured records (JSON).

For each record on the page we classify every block, then for each schema field
we pick the highest-confidence block predicted as that field. Fields whose best
confidence is below min_confidence are left null and flagged — those flags are
exactly what the monitor watches and what becomes retraining feedback.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from .config import MODELS_DIR, Config, load_config
from .crawl import Block, fetch, to_blocks
from .features import block_to_features


@lru_cache(maxsize=4)
def _load_model(tracking_uri: str, name: str):
    """Load the model for inference.

    Production deployments ship a portable `models/model.joblib` file and load
    that directly — no MLflow needed at serving time. Only if that file is
    absent (e.g. a fresh local dev checkout) do we fall back to the MLflow
    registry, importing mlflow lazily so the deployed image stays slim.
    """
    local = MODELS_DIR / "model.joblib"
    if local.exists():
        return joblib.load(local)

    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(
            f"No model file at {local} and no registered model '{name}'. "
            "Run `python -m src.train` first."
        )
    prod = [v for v in versions if v.current_stage == "Production"]
    chosen = prod[0] if prod else max(versions, key=lambda v: int(v.version))
    return mlflow.sklearn.load_model(f"models:/{name}/{chosen.version}")


def classify_blocks(blocks: list[Block], model, fields: list[str]):
    """Return list of (block, label, confidence)."""
    if not blocks:
        return []
    X = pd.DataFrame([block_to_features(b) for b in blocks])
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    results = []
    for b, row in zip(blocks, proba):
        best_idx = row.argmax()
        results.append((b, classes[best_idx], float(row[best_idx])))
    return results


def extract_from_html(html: str, url: str, cfg: Config, model=None) -> dict:
    if model is None:
        model = _load_model(cfg.mlops.mlflow_tracking_uri,
                             cfg.mlops.registered_model_name)
    blocks = to_blocks(html, url, cfg)
    scored = classify_blocks(blocks, model, cfg.schema_.field_names)

    # group by record, then take best block per field
    records: dict[int, dict] = {}
    confidences: list[float] = []
    for b, label, conf in scored:
        if label == "none":
            continue
        rec = records.setdefault(b.record_id, {})
        prev = rec.get(label)
        if prev is None or conf > prev["_conf"]:
            rec[label] = {"value": b.text, "_conf": conf}

    output_records = []
    filled, total = 0, 0
    for rid in sorted(records):
        rec_out, rec_conf = {}, {}
        for field in cfg.schema_.field_names:
            total += 1
            cell = records[rid].get(field)
            if cell and cell["_conf"] >= cfg.mlops.min_confidence:
                rec_out[field] = cell["value"]
                rec_conf[field] = round(cell["_conf"], 3)
                confidences.append(cell["_conf"])
                filled += 1
            else:
                rec_out[field] = None
                rec_conf[field] = round(cell["_conf"], 3) if cell else 0.0
        output_records.append({"fields": rec_out, "confidence": rec_conf})

    success_rate = filled / total if total else 0.0
    return {
        "url": url,
        "n_records": len(output_records),
        "success_rate": round(success_rate, 3),
        "mean_confidence": round(sum(confidences) / len(confidences), 3)
                           if confidences else 0.0,
        "records": output_records,
    }


def extract_from_url(url: str, cfg: Config) -> dict:
    html = fetch(url, cfg)
    return extract_from_html(html, url, cfg)


if __name__ == "__main__":
    c = load_config()
    result = extract_from_url(c.target.start_urls[0], c)
    print(json.dumps(result, indent=2)[:2000])
