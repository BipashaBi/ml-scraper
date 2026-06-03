.PHONY: install demo crawl data train evaluate extract serve monitor retrain test lint clean

install:
	pip install -r requirements.txt
	# Only needed for JS sites (render_js: true):
	# playwright install chromium

# One command to see the whole loop: crawl -> weak-label -> train -> promote -> extract
demo: train evaluate extract

crawl:
	python -m src.crawl

data:
	python -m src.dataset

train:
	python -m src.train

evaluate:
	python -m src.evaluate

extract:
	python -m src.extract

monitor:
	python -m src.monitor

# Closed-loop: retrain only if the monitor flags drift/perf (use --force to override)
retrain:
	python -m src.retrain

serve:
	uvicorn service.app:app --reload --port 8000

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

test:
	pytest -q

lint:
	ruff check src service tests

clean:
	rm -rf data/raw/* data/labeled/* mlruns __pycache__ */__pycache__
