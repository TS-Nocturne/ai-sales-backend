# AI Sales Brain

LangGraph sales agent for mobile-accessory retail — product search (Pinecone RAG), discount negotiation, lead scoring, HITL manager approval, payments (Slip2Go), and LINE delivery.

Exposed as a **FastAPI brain** for the Next.js dashboard.

## Features

- **ReAct sales agent** — Gemini with `search_knowledge_base` + `calculate_discount`
- **Lead scoring** — 1–100 score + pipeline stage per turn
- **HITL discount approval** — interrupts when discount > 15%
- **Persistent memory** — Neon PostgreSQL (`PostgresSaver`) per `thread_id`
- **Payments** — slip verification + PromptPay QR
- **Knowledge indexing** — PDF/TXT/CSV → Pinecone

## Architecture

```
context_summarizer → sales_agent ⇄ tool_executor
                          ↓
                     lead_scorer → [INTERRUPT] human_approval → post_approval → END
```

## Quick Start (development)

```bash
poetry install
cp .env.example .env   # set DATABASE_URL, GEMINI_API_KEY, etc.
```

### HTTP API (production target)

```bash
poetry run python -m ai_sales serve --host 0.0.0.0 --port 8000
# or
make serve
```

### CLI

```bash
poetry run python -m ai_sales chat
poetry run python -m ai_sales demo
```

### Tests

```bash
poetry run pytest
make test
```

## Deploy

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the full checklist.

```bash
# Docker (production-like)
docker compose up --build

# Or build manually
docker build -t ai-sales-brain .
docker run --rm -p 8000:8000 --env-file .env -e ENV=production ai-sales-brain
```

Health probes: `GET /health` (liveness), `GET /health/ready` (readiness).

## API (summary)

All protected routes require header `X-Brain-Key: <BRAIN_API_KEY>`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `GET` | `/health/ready` | Readiness |
| `POST` | `/chat` | Customer message → agent reply + state |
| `POST` | `/chat/async` | Async chat (LINE / webhook) |
| `POST` | `/approval` | Manager HITL approve/reject |
| `GET` | `/state/{thread_id}` | Conversation snapshot |
| `POST` | `/payments/*` | Slip verify, PromptPay QR |
| `POST` | `/knowledge/*` | Index / delete documents |

## Environment Variables

See `.env.example`. Minimum for production (`ENV=production`):

| Variable | Required |
|----------|----------|
| `DATABASE_URL` | Yes (Neon **pooler** URL) |
| `GEMINI_API_KEY` | Yes |
| `BRAIN_API_KEY` | Yes |
| `INTERNAL_API_KEY` | Yes |
| `DASHBOARD_URL` | Yes |
| `PINECONE_API_KEY` | Recommended |
| `CATALOG_API_URL` | Recommended |

## Project Structure

```
ai_sales/
├── api/             # FastAPI server + service layer
├── channels/        # LINE delivery
├── config/          # LLM, prompts, vectorstore
├── graph/           # StateGraph builder (PostgresSaver)
├── knowledge/       # Document indexing
├── nodes/           # Agent, scorer, HITL nodes
├── payments/        # Slip2Go, PromptPay QR
├── runtime.py       # Shared Postgres pool for CLI/demo
├── tools/           # Catalog, sales tools
├── cli.py           # Interactive chat
└── main.py          # HITL demo
```

## License

MIT
