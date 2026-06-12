FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.5 \
    && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --only main

COPY ai_sales ./ai_sales

ENV PYTHONUNBUFFERED=1 \
    ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "ai_sales.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
