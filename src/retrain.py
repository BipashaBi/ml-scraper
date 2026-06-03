"""Closed-loop retraining.

Glue that runs on a schedule: monitor -> (if drift/perf alert) train challenger
-> evaluate against champion -> promote if better. `--force` retrains regardless,
which is what CI uses for a smoke test.
"""
from __future__ import annotations

import argparse

from .config import load_config
from .evaluate import promote_if_better
from .monitor import monitor
from .train import train


def run(force: bool = False) -> None:
    cfg = load_config()
    if not force:
        report = monitor(cfg)
        if not report["needs_retrain"]:
            print("[retrain] no retrain needed; champion stays.")
            return
        print("[retrain] alert detected -> training challenger")
    train(cfg, rebuild_data=True)
    print("[retrain]", promote_if_better(cfg))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(**vars(ap.parse_args()))
