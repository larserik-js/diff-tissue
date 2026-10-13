.PHONY: format lint type test-full check

format:
	uv run ruff format --check .

lint:
	uv run ruff check .

type:
	uv run mypy

test-coverage:
	uv run pytest --cov=src/diff_tissue/core --cov-fail-under=90

check: format lint type test-coverage
