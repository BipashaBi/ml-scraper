"""FastAPI service that serves the registered model.

Endpoints:
  GET  /health            liveness + which model version is live
  POST /extract           {url} -> structured records + confidence
  POST /feedback          human-verified field values -> appended to training set
  GET  /metrics           recent extraction success/confidence (for dashboards)

The /feedback endpoint is what closes the MLOps loop: corrections from real use
become gold labels that the next retrain trusts 3x over the weak labels.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import DATA_LABELED, load_config
from src.extract import _load_model, extract_from_url
from src.features import block_to_features
from src.crawl import Block

app = FastAPI(title="ML Web Scraper", version="0.1.0")
CFG = load_config()


class ExtractRequest(BaseModel):
    url: str


class FeedbackItem(BaseModel):
    text: str
    tag: str = "span"
    classes: str = ""
    label: str  # the correct field name, or "none"


class FeedbackRequest(BaseModel):
    items: list[FeedbackItem]


@app.get("/health")
def health():
    try:
        _load_model(CFG.mlops.mlflow_tracking_uri,
                    CFG.mlops.registered_model_name)
        model_ok = True
    except Exception:
        model_ok = False
    return {"status": "ok", "model_loaded": model_ok,
            "target": CFG.target.name}


@app.post("/extract")
def extract(req: ExtractRequest):
    try:
        return extract_from_url(req.url, CFG)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    valid = set(CFG.schema_.labels)
    fb_path = DATA_LABELED / f"{CFG.target.name}_feedback.jsonl"
    written = 0
    with open(fb_path, "a", encoding="utf-8") as fh:
        for item in req.items:
            if item.label not in valid:
                raise HTTPException(400, f"unknown label '{item.label}'")
            b = Block(text=item.text, tag=item.tag, classes=item.classes,
                      record_id=0, url="feedback")
            row = block_to_features(b)
            row["label"] = item.label
            fh.write(json.dumps(row) + "\n")
            written += 1
    return {"accepted": written, "stored_at": str(fb_path.name)}


@app.get("/metrics")
def metrics():
    log_path = DATA_LABELED.parent / "monitoring_log.jsonl"
    if not log_path.exists():
        return {"recent": []}
    lines = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    return {"recent": lines[-10:]}