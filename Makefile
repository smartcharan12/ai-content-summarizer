# Convenience commands (Linux/macOS). Windows users: run the command after the
# colon manually in PowerShell, e.g. `.venv\Scripts\python -m pytest`.
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install lint format test test-fast data eda smoke train models api api-demo docker-build docker-run clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

venv:            ## Create the virtual environment
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv    ## Install everything for local development (CPU torch)
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

lint:            ## Run Ruff linter
	$(PY) -m ruff check src api tests
	$(PY) -m ruff format --check src api tests

format:          ## Auto-format and auto-fix lint issues
	$(PY) -m ruff format src api tests
	$(PY) -m ruff check --fix src api tests

test:            ## Run the whole test suite with coverage
	$(PY) -m pytest --cov=summarizer --cov-report=term-missing

test-fast:       ## Skip slow tests (model downloads / real inference)
	$(PY) -m pytest -m "not slow"

data:            ## Stages 1-2: download SAMSum and validate it
	$(PY) -m summarizer.pipeline.training_pipeline --stages ingestion validation

eda:             ## Re-execute the EDA notebook in place (needs `make data` first)
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb

smoke:           ## Full pipeline with a tiny random model (CPU, ~1 min; outputs in artifacts/smoke/)
	SMOKE_TEST=1 $(PY) -m summarizer.pipeline.training_pipeline

train:           ## Real fine-tuning: needs a GPU (see notebooks/train_on_colab.ipynb)
	$(PY) -m summarizer.pipeline.training_pipeline --stages transformation training

models:          ## List registered model versions
	$(PY) -m summarizer.components.model_registry list

api:             ## Serve the API + UI (model from the registry / MODEL_NAME_OR_PATH)
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

api-demo:        ## Serve with the 1.6M-param random model (low-RAM laptops; output is gibberish)
	MODEL_SOURCE=path MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-PegasusForConditionalGeneration \
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

docker-build:    ## Build the inference image
	docker build -t ai-content-summarizer .

docker-run:      ## Run the image on :8000 (first start downloads the model)
	docker run --rm -p 8000:8000 -v hf-cache:/app/.cache/huggingface ai-content-summarizer

clean:           ## Remove caches (keeps data and models)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
