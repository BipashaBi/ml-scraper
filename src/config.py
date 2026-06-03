"""Typed configuration loaded from config/config.yaml.

Centralising config in a validated object (rather than passing dicts around)
is a small thing that interviewers notice: it makes the pipeline reproducible
and fails loudly on a bad config instead of silently mis-scraping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"


class Field_(BaseModel):
    name: str
    hints: dict[str, Any] = Field(default_factory=dict)


class Schema(BaseModel):
    fields: list[Field_]

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def labels(self) -> list[str]:
        # "none" is the catch-all class for blocks that are not a target field.
        return self.field_names + ["none"]


class Target(BaseModel):
    name: str
    start_urls: list[str]
    max_pages: int = 3
    record_selector: str
    render_js: bool = False
    render_engine: str = "playwright"   # playwright | selenium
    request_delay_seconds: float = 1.0
    user_agent: str = "ml-scraper/0.1"


class MLOps(BaseModel):
    mlflow_tracking_uri: str = "file:./mlruns"
    registered_model_name: str = "block-field-classifier"
    min_confidence: float = 0.55
    promote_metric: str = "macro_f1"
    promote_min_delta: float = 0.01
    drift_psi_threshold: float = 0.2
    min_success_rate: float = 0.80


class Config(BaseModel):
    target: Target
    schema_: Schema = Field(alias="schema")
    mlops: MLOps

    model_config = {"populate_by_name": True}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)


# Convenient paths used across the project.
DATA_RAW = ROOT / "data" / "raw"
DATA_LABELED = ROOT / "data" / "labeled"
MODELS_DIR = ROOT / "models"
for _d in (DATA_RAW, DATA_LABELED, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
