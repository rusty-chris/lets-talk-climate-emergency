# Shared image for the `api` and `ui` docker-compose stub services.
# Each service overrides `command:` in docker-compose.yml; `qdrant` uses the
# official upstream image directly and has no Dockerfile of its own.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
