FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

ENV UV_CACHE_DIR=/tmp/uv-cache

RUN uv sync --frozen --no-dev && rm -rf /tmp/uv-cache

COPY src ./src

EXPOSE 8000

CMD ["sh", "-c", "uv run python -m uvicorn diff_tissue.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
