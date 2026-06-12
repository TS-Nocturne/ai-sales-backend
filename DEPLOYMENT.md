# AI-Sales Brain — Deployment Guide

Production surface is the **FastAPI brain** (`ai-sales serve`), called by the Next.js dashboard over HTTP.

## Architecture

```
Next.js Dashboard
    │  X-Brain-Key
    ▼
FastAPI Brain (this repo) ──► Neon PostgreSQL (LangGraph checkpoints)
    │                         Pinecone (product + FAQ vectors)
    ├─► Google Gemini (LLM + embeddings)
    ├─► Slip2Go (slip verify + PromptPay QR)
    └─► Dashboard callbacks (orders / messages)
```

## Prerequisites

| Service | Purpose |
|---------|---------|
| **Neon PostgreSQL** | `PostgresSaver` — use **pooled** URL (`-pooler` in hostname) |
| **Google Gemini** | `GEMINI_API_KEY` |
| **Pinecone** | Vector search (recommended for production) |
| **Next.js Dashboard** | `DASHBOARD_URL`, `CATALOG_API_URL`, `INTERNAL_API_KEY` |

## Environment Variables

Copy and fill:

```bash
cp .env.example .env
```

Set `ENV=production` on the server. Required in production:

- `DATABASE_URL` — Neon **pooler** URL
- `GEMINI_API_KEY`
- `BRAIN_API_KEY` — dashboard → brain auth (`X-Brain-Key`)
- `INTERNAL_API_KEY` — brain → dashboard callbacks
- `DASHBOARD_URL`

See `.env.example` for the full list (LINE, Slip2Go, Pinecone, CORS, etc.).

## First Deploy Checklist

1. Create Neon database; copy **pooled** `DATABASE_URL`
2. Set all secrets in your platform (never bake into the image)
3. Set `ENV=production`
4. Deploy the brain container / process
5. Verify health:
   - `GET /health` — liveness
   - `GET /health/ready` — readiness (graph + env)
6. Point Next.js `BRAIN_URL` to this service with matching `BRAIN_API_KEY`
7. (Optional) Seed Pinecone: `poetry run python -m ai_sales.tools.seed_pinecone`

`PostgresSaver.setup()` runs automatically on first graph build — no manual migration step.

## Run Locally (production-like)

```bash
poetry install
cp .env.example .env   # fill secrets

# Docker
docker compose up --build

# Or native
ENV=production poetry run python -m ai_sales serve --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t ai-sales-brain .
docker run --rm -p 8000:8000 --env-file .env ai-sales-brain
```

Image entrypoint:

```bash
uvicorn ai_sales.api.server:app --host 0.0.0.0 --port 8000
```

## API Authentication

Dashboard requests must include:

```
X-Brain-Key: <BRAIN_API_KEY>
```

## Health Probes

| Path | Use |
|------|-----|
| `GET /health` | Liveness — process is up |
| `GET /health/ready` | Readiness — graph + required env |

Configure your load balancer / platform to use `/health/ready` for traffic routing.

## Scaling Notes

- Graph singleton + per-thread locks: **one brain instance** is simplest
- Horizontal scaling requires sticky sessions per `thread_id` or an external job queue for `/chat/async`
- Keep `DATABASE_POOL_MAX_SIZE` small (5–10) when using Neon pooler

## CLI / Demo (development)

Requires `DATABASE_URL` in `.env`:

```bash
poetry run python -m ai_sales chat
poetry run python -m ai_sales demo
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `DATABASE_URL environment variable is required` | Set Neon pooled URL |
| Neon connection limit errors | Use `-pooler` URL; lower `DATABASE_POOL_MAX_SIZE` |
| `Missing required environment variables for production` | Set `BRAIN_API_KEY`, `INTERNAL_API_KEY`, `GEMINI_API_KEY`, `DASHBOARD_URL` |
| Docker build fails | Ensure `poetry.lock` is committed; no missing COPY paths |
