"""Minimal but real tests. CI runs these on every push."""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.crawl import Block, to_blocks
from src.dataset import label_blocks
from src.features import block_to_features
from src.train import build_pipeline

SAMPLE_HTML = """
<html><body>
  <article class="product_pod">
    <h3><a title="A Light in the Attic">A Light in the Attic</a></h3>
    <p class="price_color">£51.77</p>
    <p class="instock availability">In stock</p>
    <p class="star-rating Three"></p>
  </article>
</body></html>
"""


def _cfg():
    return load_config()


def test_blocks_extracted():
    cfg = _cfg()
    blocks = to_blocks(SAMPLE_HTML, "http://t", cfg)
    assert any("£51.77" in b.text for b in blocks)


def test_weak_labels_cover_fields():
    cfg = _cfg()
    blocks = to_blocks(SAMPLE_HTML, "http://t", cfg)
    rows = label_blocks(blocks, cfg)
    labels = {r["label"] for r in rows}
    # price hint should fire on the currency block
    assert "price" in labels


def test_features_have_signal():
    b = Block(text="£51.77", tag="p", classes="price_color",
              record_id=0, url="x")
    f = block_to_features(b)
    assert f["has_currency"] == 1.0
    assert f["n_digits"] == 4.0


def test_pipeline_fits_and_predicts():
    rows = [
        {"text": "£51.77", "classes": "price_color", "tag": "p", "length": 6,
         "n_digits": 4, "digit_ratio": 0.6, "has_currency": 1, "n_words": 1,
         "label": "price"},
        {"text": "A Light in the Attic", "classes": "", "tag": "a", "length": 20,
         "n_digits": 0, "digit_ratio": 0, "has_currency": 0, "n_words": 5,
         "label": "title"},
        {"text": "In stock", "classes": "instock availability", "tag": "p",
         "length": 8, "n_digits": 0, "digit_ratio": 0, "has_currency": 0,
         "n_words": 2, "label": "availability"},
        {"text": "footer junk", "classes": "footer", "tag": "div", "length": 11,
         "n_digits": 0, "digit_ratio": 0, "has_currency": 0, "n_words": 2,
         "label": "none"},
    ] * 4
    df = pd.DataFrame(rows)
    pipe = build_pipeline()
    pipe.fit(df.drop(columns=["label"]), df["label"])
    pred = pipe.predict(df.drop(columns=["label"]).head(1))
    assert pred[0] in {"price", "title", "availability", "none"}
