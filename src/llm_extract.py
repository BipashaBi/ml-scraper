"""LLM-based fact extraction for a single page (e.g. a Wikipedia company article).

Unlike the classical scraper (which classifies repeating items by their content),
this asks an LLM to *read* one page and return labeled facts as JSON. That's the
right tool for single-entity, key-value extraction — and it generalizes to any
page without per-site tuning.

Usage:
    python -m src.llm_extract https://en.wikipedia.org/wiki/Eli_Lilly_and_Company
"""
from __future__ import annotations

import json
import os
import sys

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from .config import load_config
from .crawl import fetch

load_dotenv()  # reads GEMINI_API_KEY from a local .env file (never committed)

MODEL = "gemini-2.5-flash"  # swap to "gemini-2.0-flash" if this isn't available to you
DEFAULT_FIELDS = [
    "name", "founder", "founded", "headquarters",
    "industry", "revenue", "number_of_employees", "ceo", "website",
]


def page_to_text(html: str, max_chars: int = 8000) -> str:
    """Strip a page down to clean text and cap its length to keep the call cheap."""
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup(["script", "style", "noscript"]):
        junk.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    return text[:max_chars]


def extract_facts(url: str, fields: list[str] | None = None) -> dict:
    fields = fields or DEFAULT_FIELDS
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Create a .env file in the project root "
            "containing:  GEMINI_API_KEY=your_key_here"
        )

    # Lazy import so the rest of the project runs even without the LLM library.
    from google import genai

    html = fetch(url, load_config())   # reuses the robots.txt-respecting fetcher
    text = page_to_text(html)

    prompt = (
        "From the page text below, extract facts about the main entity. "
        "Return ONLY a JSON object with exactly these keys: "
        f"{', '.join(fields)}. Use null for any value not stated in the text.\n\n"
        f"PAGE TEXT:\n{text}"
    )
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return json.loads(resp.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m src.llm_extract <url>")
        raise SystemExit(1)
    facts = extract_facts(sys.argv[1])
    print(json.dumps(facts, indent=2, ensure_ascii=False))
