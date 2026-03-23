.PHONY: format lint type test-full check

format:
	uv run ruff format --check .

lint:
	uv run ruff check .

type:
	uv run mypy

test-full:
	uv run pytest

check: format lint type test-full
