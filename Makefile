.PHONY: install install-ml run api test lint format clean

PY ?= python
PIP ?= pip
UVICORN ?= uvicorn

install:            ## Install core runtime dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-ml:         ## Install optional GPU/ML training stack (Linux + CUDA)
	$(PIP) install -r requirements-ml.txt

run:                ## Run the full CLI pipeline (data -> SFT -> DPO -> inference)
	$(PY) scripts/run_pipeline.py

api:                ## Launch the FastAPI REST server
	$(UVICORN) src.api:app --reload --host 0.0.0.0 --port 8000

test:               ## Run the pytest suite
	$(PY) -m pytest

lint:               ## Run ruff linting
	$(PY) -m ruff check .

format:             ## Auto-fix lint issues with ruff
	$(PY) -m ruff check . --fix

clean:              ## Remove build artifacts and generated outputs
	rm -rf outputs .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__