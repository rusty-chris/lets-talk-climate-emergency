# Shared image for the `api` and `ui` docker-compose stub services.
# Each service overrides `command:` in docker-compose.yml; `qdrant` uses the
# official upstream image directly and has no Dockerfile of its own.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Two-step sync (the standard uv Docker pattern): install third-party deps
# from the lockfile alone first (cache-friendly layer; --no-install-project
# avoids needing README.md/packages at this stage), then copy the repo and
# install the project itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
