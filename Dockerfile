FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_PROVIDER=ollama \
    OLLAMA_HOST=http://ollama:11434

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY dsqa ./dsqa
COPY eval ./eval
COPY docker-entrypoint.sh ./docker-entrypoint.sh

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -e . \
    && chmod +x docker-entrypoint.sh

EXPOSE 8765

ENTRYPOINT ["./docker-entrypoint.sh"]
