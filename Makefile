.PHONY: help install test test-full lint format type-check clean

help:
	@echo "strands-dspy - Makefile commands"
	@echo ""
	@echo "  make install      - Install package with uv"
	@echo "  make test         - Run unit tests"
	@echo "  make test-full    - Run all tests including integration"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code with black"
	@echo "  make type-check   - Run type checking"
	@echo "  make clean        - Clean build artifacts"
	@echo ""

install:
	@echo "Installing strands-dspy with uv..."
	uv pip install -e ".[dev]"

test:
	@echo "Running unit tests..."
	uv run pytest tests/test_*.py -v --cov=strands_dspy --cov-report=term-missing

test-full:
	@echo "Running all tests..."
	./run_tests.sh --full

lint:
	@echo "Running linting..."
	uv run ruff check src tests examples

format:
	@echo "Formatting code..."
	uv run black src tests examples

type-check:
	@echo "Running type checks..."
	uv run mypy src

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
