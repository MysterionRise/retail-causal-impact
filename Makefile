# path: Makefile
.PHONY: help setup data train report full app api web dev test lint clean all

help:
	@echo "Coupon Causal Impact Analysis - Available Commands:"
	@echo "  make setup    - Install dependencies and create directories"
	@echo "  make data     - Generate/load data"
	@echo "  make train    - Run causal estimation pipeline"
	@echo "  make report   - Generate reports and figures"
	@echo "  make full     - Run data → train → report with the heavy config/full.yaml profile"
	@echo "  make app      - Launch Streamlit dashboard"
	@echo "  make api      - Launch the FastAPI causal engine + AG-UI copilot endpoint (:8000)"
	@echo "  make web      - Launch the Next.js + CopilotKit front end (:3000)"
	@echo "  make dev      - Launch api and web together"
	@echo "  make test     - Run test suite"
	@echo "  make lint     - Run code quality checks"
	@echo "  make clean    - Remove generated files"
	@echo "  make all      - Run pipeline with the fast default profile (setup → data → train → report)"

setup:
	@echo "Setting up environment..."
	pip install -e ".[dev]"
	@mkdir -p data/raw data/interim data/processed
	@mkdir -p reports/figures reports/tables
	@mkdir -p models
	@echo "Setup complete!"

data:
	@echo "Generating synthetic coupon campaign data..."
	python -m scripts.run_pipeline --config config/default.yaml --data_source synth --stage data
	@echo "Data generation complete!"

train:
	@echo "Running causal estimation pipeline..."
	python -m scripts.run_pipeline --config config/default.yaml --data_source synth --stage estimation
	@echo "Training complete!"

report:
	@echo "Generating reports..."
	python -m scripts.run_pipeline --config config/default.yaml --data_source synth --stage report
	@echo "Reports saved to reports/"

full:
	@echo "Running full-profile pipeline (this takes a while)..."
	python -m scripts.run_pipeline --config config/full.yaml --data_source synth --stage all
	@echo "Full pipeline complete! Reports saved to reports/"

app:
	@echo "Launching Streamlit dashboard..."
	streamlit run app/streamlit_app.py

api:
	@test -f models/estimation_results.joblib || (echo "No pipeline artifacts yet; running make all first" && $(MAKE) data train report)
	@echo "Launching causal engine API on http://localhost:8000 ..."
	uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir services --reload-dir src

web:
	@test -d web/node_modules || (cd web && npm install)
	@echo "Launching CopilotKit front end on http://localhost:3000 ..."
	cd web && npm run dev

dev:
	@$(MAKE) -j2 api web

test:
	@echo "Running test suite..."
	pytest tests/ services/api/tests -v --cov=src/coupon_causal --cov=services/api --cov-report=term-missing
	@echo "Tests complete!"

lint:
	@echo "Running code quality checks..."
	ruff check src/ tests/ scripts/ app/ services/
	mypy src/coupon_causal/
	@echo "Linting complete!"

clean:
	@echo "Cleaning generated files..."
	rm -rf data/interim/* data/processed/*
	rm -rf reports/figures/* reports/tables/*
	rm -rf models/*
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

all: setup data train report
	@echo "Full pipeline complete!"
