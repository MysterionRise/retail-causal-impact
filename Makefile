# path: Makefile
.PHONY: help setup data train report app test lint clean all

help:
	@echo "Coupon Causal Impact Analysis - Available Commands:"
	@echo "  make setup    - Install dependencies and create directories"
	@echo "  make data     - Generate/load data"
	@echo "  make train    - Run causal estimation pipeline"
	@echo "  make report   - Generate reports and figures"
	@echo "  make app      - Launch Streamlit dashboard"
	@echo "  make test     - Run test suite"
	@echo "  make lint     - Run code quality checks"
	@echo "  make clean    - Remove generated files"
	@echo "  make all      - Run full pipeline (setup → data → train → report)"

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

app:
	@echo "Launching Streamlit dashboard..."
	streamlit run app/streamlit_app.py

test:
	@echo "Running test suite..."
	pytest tests/ -v --cov=src/coupon_causal --cov-report=term-missing
	@echo "Tests complete!"

lint:
	@echo "Running code quality checks..."
	ruff check src/ tests/ scripts/
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
