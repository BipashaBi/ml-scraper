"""Retrieval + block-extraction layer.

Two responsibilities, deliberately kept separate from the ML:
  1. fetch()    -> get rendered HTML for a URL (HTTP or Playwright)
  2. to_blocks() -> turn a page into a flat list of candidate "blocks"

A *block* is one visible chunk of a record (a tag with its text + structural
metadata). The model later classifies each block into one of the schema fields
or "none". Keeping retrieval dumb and the intelligence in the model is what makes
this resilient to layout changes — the whole point of the ML approach.
"""
from __future__ import annotations

import json
import time
import urllib.robotparser as robotparser
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from .config import DATA_RAW, Config


@dataclass
class Block:
    """One candidate element inside a record."""
    text: str
    tag: str
    classes: str          # space-joined class tokens (a strong structural signal)
    record_id: int        # which record on the page this came from
    url: str

    # cheap engineered features (computed once, reused by the model)
    @property
    def length(self) -> int:
        return len(self.text)

    @property
    def n_digits(self) -> int:
        return sum(c.isdigit() for c in self.text)

    @property
    def has_currency(self) -> bool:
        return any(sym in self.text for sym in ("£", "$", "€", "₹", "¥"))


def _robots_allowed(url: str, user_agent: str) -> bool:
    """Respect robots.txt — being a good citizen is also a great interview answer.

    We fetch robots.txt with our *own* User-Agent. Some sites (e.g. Wikimedia)
    return 403 to default library agents, and Python's RobotFileParser treats a
    403 on robots.txt as 'disallow everything', which would wrongly block us.
    """
    try:
        parts = urlparse(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        resp = requests.get(robots_url, headers={"User-Agent": user_agent},
                            timeout=10)
        if resp.status_code >= 400:
            return True  # no usable robots.txt -> default to allowing
        rp = robotparser.RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If robots.txt is unreachable, default to caution but don't hard-crash.
        return True


def fetch(url: str, cfg: Config) -> str:
    """Return HTML for a URL. Uses Playwright only when render_js is set."""
    if not _robots_allowed(url, cfg.target.user_agent):
        raise PermissionError(f"robots.txt disallows fetching {url}")

    if cfg.target.render_js:
        engine = cfg.target.render_engine.lower()
        if engine == "selenium":
            return _fetch_selenium(url, cfg)
        return _fetch_playwright(url, cfg)
    resp = requests.get(
        url, headers={"User-Agent": cfg.target.user_agent}, timeout=30
    )
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


def _fetch_playwright(url: str, cfg: Config) -> str:
    """Lazy import so the project still runs for static sites without Playwright."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=cfg.target.user_agent)
        page.goto(url, wait_until="networkidle", timeout=45_000)
        html = page.content()
        browser.close()
    return html


def _fetch_selenium(url: str, cfg: Config) -> str:
    """Selenium fallback — the engine the CodeWithHarry tutorial demonstrates.
    Lazy import so the project still runs without Selenium installed."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--user-agent={cfg.target.user_agent}")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.set_page_load_timeout(45)
        driver.get(url)
        html = driver.page_source
    finally:
        driver.quit()
    return html


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def to_blocks(html: str, url: str, cfg: Config) -> list[Block]:
    """Strip noise, find records, and emit one Block per meaningful element."""
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup(["script", "style", "noscript", "svg"]):
        junk.decompose()

    blocks: list[Block] = []
    records = soup.select(cfg.target.record_selector)
    for rid, record in enumerate(records):
        for el in record.find_all(True):           # every descendant tag
            if not isinstance(el, Tag):
                continue
            text = _clean(el.get_text(" ", strip=True))
            classes = " ".join(el.get("class", []))
            # Some values live only in the class (e.g. "star-rating Three").
            # Keep those: fall back to the class string as the block's value.
            if not text and classes:
                text = classes
            if not text or len(text) > 400:
                continue
            # also fold in attribute hints that carry signal (e.g. rating classes)
            blocks.append(
                Block(text=text, tag=el.name, classes=classes,
                      record_id=rid, url=url)
            )
    return blocks


def discover_links(html: str, base_url: str, cfg: Config) -> list[str]:
    """Follow in-domain pagination/detail links up to max_pages (simple BFS seed)."""
    soup = BeautifulSoup(html, "html.parser")
    domain = urlparse(base_url).netloc
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        nxt = urljoin(base_url, a["href"])
        if urlparse(nxt).netloc == domain:
            out.append(nxt)
    return out


def crawl(cfg: Config) -> list[Block]:
    """Crawl up to max_pages and return all blocks. Caches raw HTML to data/raw."""
    seen: set[str] = set()
    frontier: list[str] = list(cfg.target.start_urls)
    all_blocks: list[Block] = []

    while frontier and len(seen) < cfg.target.max_pages:
        url = frontier.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch(url, cfg)
        except Exception as exc:  # keep crawling even if one page fails
            print(f"[crawl] skip {url}: {exc}")
            continue

        # cache raw html (useful for reproducible re-labeling / debugging)
        cache = DATA_RAW / cfg.target.name
        cache.mkdir(parents=True, exist_ok=True)
        fname = cache / (str(abs(hash(url))) + ".html")
        fname.write_text(html, encoding="utf-8")

        all_blocks.extend(to_blocks(html, url, cfg))
        for link in discover_links(html, url, cfg):
            if link not in seen and len(seen) + len(frontier) < cfg.target.max_pages:
                frontier.append(link)
        time.sleep(cfg.target.request_delay_seconds)

    print(f"[crawl] {len(seen)} pages -> {len(all_blocks)} blocks")
    return all_blocks


def blocks_to_jsonl(blocks: list[Block], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for b in blocks:
            fh.write(json.dumps(asdict(b)) + "\n")


if __name__ == "__main__":
    from .config import load_config
    c = load_config()
    bs = crawl(c)
    blocks_to_jsonl(bs, DATA_RAW / f"{c.target.name}_blocks.jsonl")
    print(f"saved {len(bs)} blocks")
