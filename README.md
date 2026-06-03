# Self-Healing ML Web Scraper (with full MLOps loop)

A web scraper that learns *what* a field looks like instead of hard-coding *where*
it is. Traditional scrapers break the moment a site renames a CSS class. This one
classifies page blocks by **content + structure**, so it survives layout changes —
and when accuracy finally drifts, it **detects that itself and retrains**.

Point it at any site by editing one YAML file. It ships wired to a legal scraping
sandbox (`books.toscrape.com`) so it runs end-to-end out of the box.

---

## Why this is different from a normal scraper

| Traditional scraper | This project |
|---|---|
| Hard-coded XPath/CSS selectors | Learns fields from text + structure |
| Breaks on layout change | Resilient; flags low-confidence fields |
| Manual fixes when it breaks | Monitors drift and retrains automatically |
| No labeled data needed, but brittle | Bootstraps labels via weak supervision, improves on feedback |

---

## Architecture

```
                          config/config.yaml  (the only file you edit)
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                            ▼
  RETRIEVAL                    TRAINING                     SERVING
  src/crawl.py             src/dataset.py  ──► weak supervision (labeling fns)
  fetch + robots.txt       src/train.py    ──► sklearn pipeline + MLflow tracking
  → candidate blocks       src/evaluate.py ──► champion/challenger promotion
        │                          │                            │
        └─────────► src/extract.py (model → structured JSON) ◄──┘
                                   │                     service/app.py (FastAPI)
                                   ▼                     /extract /feedback /metrics
                          src/monitor.py  ──► success rate + PSI drift
                                   │
                          src/retrain.py  ──► closed loop: monitor→train→promote
```

**The MLOps loop in one line:** crawl → weak-label → train (tracked) → gate-promote
→ serve → monitor (drift + success) → collect feedback → retrain → re-promote.

---

## Quickstart

```bash
pip install -r requirements.txt        # add `playwright install chromium` only for JS sites
make demo                              # crawl → label → train → promote → print JSON
make serve                             # FastAPI on :8000
make mlflow-ui                         # experiment tracking UI on :5000
make monitor                           # drift + success-rate check
make retrain                           # retrains ONLY if monitor flags a problem
```

Try the API:
```bash
curl -X POST localhost:8000/extract -H 'Content-Type: application/json' \
  -d '{"url":"https://books.toscrape.com/catalogue/page-1.html"}'
```

Send a correction (this becomes a gold label, trusted 3× at next retrain):
```bash
curl -X POST localhost:8000/feedback -H 'Content-Type: application/json' \
  -d '{"items":[{"text":"£51.77","classes":"price_color","tag":"p","label":"price"}]}'
```

---

## Point it at YOUR site

Edit `config/config.yaml`:
1. `target.start_urls` and `record_selector` (the repeating container, e.g. a product card).
2. `schema.fields` — list the fields you want, with optional `hints` (labeling clues).
3. Set `render_js: true` if the site needs a browser, and `render_engine: playwright` or `selenium`.
4. Run `make demo`.

No Python changes needed. The hints get the first model off the ground with zero
manual labeling; real usage + `/feedback` make it better over time.

---

## Key engineering decisions (talking points)

- **Weak supervision** (`src/dataset.py`): config hints compile into labeling
  functions that majority-vote noisy labels — a lightweight Snorkel. Avoids the
  cold-start "I have no labeled data" problem.
- **Structure as features** (`src/features.py`): currency symbols, digit ratios,
  tag, and class char-n-grams. This is *why* it's resilient to HTML churn.
- **Confidence-gated extraction** (`src/extract.py`): low-confidence fields are
  left null and flagged, not guessed — the signal that powers monitoring.
- **Champion/challenger** (`src/evaluate.py`): a new model only ships if it beats
  production on macro-F1 by a margin. No silent regressions.
- **Drift detection via PSI** (`src/monitor.py`): compares live page features to a
  baseline saved at training time; closes the loop into `src/retrain.py`.
- **Reproducibility**: every run tracked in MLflow; models in the registry with
  Production/Archived stages for one-command rollback.

---

## Resume bullets (steal these)

- Built a **self-healing ML web scraper** that classifies page elements by content
  and structure, achieving site-agnostic extraction resilient to HTML layout changes.
- Designed a **weak-supervision** labeling pipeline (Snorkel-style labeling functions)
  to bootstrap a training set with **zero manual annotation**.
- Implemented an end-to-end **MLOps loop** — MLflow experiment tracking + model
  registry, **champion/challenger** promotion gating, **PSI-based drift monitoring**,
  and automated retraining triggered by live performance signals.
- Shipped the model behind a **FastAPI** service (Dockerized) with a human-in-the-loop
  **/feedback** endpoint that converts corrections into gold labels for retraining.
- Set up **CI** (lint + tests + smoke-train) so every change is validated automatically.

---

## Tech stack
Python · scikit-learn · requests · BeautifulSoup · Playwright / Selenium · MLflow · FastAPI · Docker · GitHub Actions

## Notes & honest limitations
- The shipped config targets a sandbox built for scraping practice. **Respect
  robots.txt and each site's Terms of Service** before pointing it elsewhere.
- The classical-ML model is intentionally small/CPU-only. For messier sites you can
  swap the estimator in `src/train.py:build_pipeline()` (e.g. a fine-tuned
  transformer/NER model) without touching the rest of the loop — that's the point of
  the modular design.
- Anti-bot evasion (CAPTCHAs, IP bans) is out of scope; add proxies/Playwright stealth
  if a target requires it.
