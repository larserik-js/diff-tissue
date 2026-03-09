.PHONY: format lint type check

format:
	uv run ruff format --check .

lint:
	uv run ruff check .

type:
	uv run mypy

check: format lint type
