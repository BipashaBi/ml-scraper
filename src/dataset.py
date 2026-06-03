"""Build a labeled training set from crawled blocks via WEAK SUPERVISION.

This is the trick that makes "train your own model" practical with zero manual
labeling. We write small, noisy *labeling functions* (LFs) derived from the
`hints` in config.yaml. Each LF votes a field label for a block; we take a
confident majority vote. The resulting noisy labels train a model that then
*generalizes beyond the rules* — and the retrain loop replaces these heuristic
labels with real human-verified feedback over time.

This is the same idea behind Snorkel, kept deliberately lightweight.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import DATA_LABELED, Config
from .crawl import Block, crawl
from .features import block_to_features

ABSTAIN = "__abstain__"


def _make_labeling_functions(cfg: Config):
    """Compile config `hints` into callables: Block -> label | ABSTAIN."""
    lfs = []
    for field in cfg.schema_.fields:
        h = field.hints
        name = field.name

        def lf(b: Block, h=h, name=name) -> str:
            txt = b.text.lower()
            if "tag_in" in h and b.tag not in h["tag_in"]:
                return ABSTAIN
            if "min_len" in h and b.length < h["min_len"]:
                return ABSTAIN
            if "max_len" in h and b.length > h["max_len"]:
                return ABSTAIN
            if h.get("has_digits") and b.n_digits == 0:
                return ABSTAIN
            if h.get("contains_currency") and not b.has_currency:
                return ABSTAIN
            if "keywords" in h:
                hay = (txt + " " + b.classes.lower())
                if not any(k.lower() in hay for k in h["keywords"]):
                    return ABSTAIN
            return name

        lfs.append(lf)
    return lfs


def label_blocks(blocks: list[Block], cfg: Config) -> list[dict]:
    """Apply LFs, majority-vote, and emit feature dicts with a `label`."""
    lfs = _make_labeling_functions(cfg)
    rows: list[dict] = []
    for b in blocks:
        votes = [v for v in (lf(b) for lf in lfs) if v != ABSTAIN]
        if votes:
            # majority vote; ties resolved by first-seen (rare in practice)
            label = Counter(votes).most_common(1)[0][0]
        else:
            label = "none"
        feats = block_to_features(b)
        feats["label"] = label
        rows.append(feats)
    return rows


def build_dataset(cfg: Config, merge_feedback: bool = True) -> Path:
    """Crawl -> weak-label -> (optionally) merge human feedback -> save jsonl."""
    blocks = crawl(cfg)
    rows = label_blocks(blocks, cfg)

    if merge_feedback:
        fb = DATA_LABELED / f"{cfg.target.name}_feedback.jsonl"
        if fb.exists():
            with open(fb, encoding="utf-8") as fh:
                fb_rows = [json.loads(line) for line in fh if line.strip()]
            # human-verified rows are gold: they overwrite, and we upweight by
            # duplicating them so the model trusts them more than weak labels.
            rows.extend(fb_rows * 3)
            print(f"[dataset] merged {len(fb_rows)} feedback rows (x3 weight)")

    out = DATA_LABELED / f"{cfg.target.name}_train.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    dist = Counter(r["label"] for r in rows)
    print(f"[dataset] {len(rows)} rows -> {out.name} | label dist: {dict(dist)}")
    return out


if __name__ == "__main__":
    from .config import load_config
    build_dataset(load_config())
