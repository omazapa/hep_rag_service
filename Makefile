# Makefile for HEP RAG Service

.PHONY: help install install-dev format lint check fix quality test clean

help:
	@echo "HEP RAG Service - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install package in production mode"
	@echo "  make install-dev    - Install package with dev dependencies"
	@echo "  make format         - Format code with black and isort"
	@echo "  make lint           - Check code quality with flake8"
	@echo "  make check          - Run autoflake to check for unused imports/variables"
	@echo "  make fix            - Auto-fix code issues with autoflake"
	@echo "  make quality        - Run all quality checks (fix + format + lint)"
	@echo "  make test           - Run tests with pytest"
	@echo "  make clean          - Remove build artifacts and cache files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

format:
	@echo "Running black..."
	black etl/
	@echo "Running isort..."
	isort etl/

lint:
	@echo "Running flake8..."
	flake8 etl/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 etl/ --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

check:
	@echo "Checking for unused imports and variables with autoflake..."
	autoflake --check --recursive --remove-all-unused-imports --remove-unused-variables etl/

fix:
	@echo "Fixing unused imports and variables with autoflake..."
	autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys etl/

quality: fix format lint
	@echo "All quality checks passed!"

test:
	pytest etl/root/tests/ etl/geant4/tests/ -v --cov=etl --cov-report=term-missing

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
