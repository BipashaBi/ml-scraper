"""Featurize a Block into something the model can learn from.

We combine the raw text (TF-IDF inside the model) with cheap *structural*
features. Structure is what makes a scraper resilient: a price is a short
string with a currency symbol and digits, regardless of the surrounding HTML.
"""
from __future__ import annotations

from .crawl import Block


def block_to_features(b: Block) -> dict:
    """Return a flat feature dict. `text` and `classes` get vectorized;
    the numeric/boolean fields are used as-is by the model pipeline."""
    return {
        "text": b.text,
        "classes": b.classes or "noclass",
        "tag": b.tag,
        "length": float(b.length),
        "n_digits": float(b.n_digits),
        "digit_ratio": float(b.n_digits) / max(b.length, 1),
        "has_currency": float(b.has_currency),
        "n_words": float(len(b.text.split())),
    }


NUMERIC_FEATURES = ["length", "n_digits", "digit_ratio", "has_currency", "n_words"]
TEXT_FEATURE = "text"
CLASS_FEATURE = "classes"
TAG_FEATURE = "tag"
