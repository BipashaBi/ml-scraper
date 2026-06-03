"""Monitoring + retrain decision.

Two production signals, both cheap and interview-friendly:
  1. Extraction success rate  -> are we still filling fields confidently?
  2. Feature drift (PSI)       -> has the page's structure shifted vs training?

When either crosses the configured threshold we recommend a retrain. In a real
deployment monitor() runs on a schedule (cron / Prefect / Airflow) over recently
logged predictions; here it can also be run ad hoc against live pages.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from .config import DATA_LABELED, Config, load_config
from .crawl import fetch, to_blocks
from .extract import _load_model, extract_from_html
from .features import NUMERIC_FEATURES, block_to_features

LOG_PATH = DATA_LABELED.parent / "monitoring_log.jsonl"


def _log(event: dict) -> None:
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def _psi(expected_mean: float, expected_std: float, actual: np.ndarray,
         bins: int = 10) -> float:
    """Population Stability Index between a baseline (mean/std) and a sample.
    We bucket the standardized actual values against a standard normal grid."""
    if len(actual) < 2 or expected_std == 0:
        return 0.0
    z = (actual - expected_mean) / expected_std
    edges = np.linspace(-3, 3, bins + 1)
    act_hist, _ = np.histogram(z, bins=edges)
    act_pct = act_hist / max(act_hist.sum(), 1)
    # expected: standard-normal mass per bin
    exp_pct = np.diff(_normal_cdf(edges))
    act_pct = np.clip(act_pct, 1e-4, None)
    exp_pct = np.clip(exp_pct, 1e-4, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    from math import erf, sqrt
    return np.array([0.5 * (1 + erf(v / sqrt(2))) for v in x])


def _load_baseline(cfg: Config) -> dict:
    client = MlflowClient(tracking_uri=cfg.mlops.mlflow_tracking_uri)
    name = cfg.mlops.registered_model_name
    versions = client.search_model_versions(f"name='{name}'")
    prod = [v for v in versions if v.current_stage == "Production"] or versions
    if not prod:
        return {}
    run_id = max(prod, key=lambda v: int(v.version)).run_id
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="feature_baseline.json")
    return json.loads(Path(local).read_text())


def monitor(cfg: Config, urls: list[str] | None = None) -> dict:
    urls = urls or cfg.target.start_urls
    model = _load_model(cfg.mlops.mlflow_tracking_uri,
                        cfg.mlops.registered_model_name)
    baseline = _load_baseline(cfg)

    success_rates: list[float] = []
    feat_rows: list[dict] = []
    for url in urls:
        html = fetch(url, cfg)
        result = extract_from_html(html, url, cfg, model=model)
        success_rates.append(result["success_rate"])
        for b in to_blocks(html, url, cfg):
            feat_rows.append(block_to_features(b))

    feats = pd.DataFrame(feat_rows)
    psi = {}
    if baseline and not feats.empty:
        for col in NUMERIC_FEATURES:
            psi[col] = round(_psi(baseline[col]["mean"], baseline[col]["std"],
                                  feats[col].to_numpy()), 3)
    max_psi = max(psi.values()) if psi else 0.0
    mean_success = float(np.mean(success_rates)) if success_rates else 0.0

    drift_alert = max_psi > cfg.mlops.drift_psi_threshold
    perf_alert = mean_success < cfg.mlops.min_success_rate
    needs_retrain = drift_alert or perf_alert

    report = {
        "mean_success_rate": round(mean_success, 3),
        "max_psi": round(max_psi, 3),
        "psi_per_feature": psi,
        "drift_alert": drift_alert,
        "perf_alert": perf_alert,
        "needs_retrain": needs_retrain,
    }
    _log(report)
    print(f"[monitor] success={mean_success:.2f} max_psi={max_psi:.2f} "
          f"-> needs_retrain={needs_retrain}")
    return report


if __name__ == "__main__":
    monitor(load_config())
