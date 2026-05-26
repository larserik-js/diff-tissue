FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY src ./src

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "diff_tissue.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
