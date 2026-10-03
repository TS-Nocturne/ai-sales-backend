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
├── heuristics.py    # Gemini workarounds (see below)
├── knowledge/       # Document indexing
├── nodes/           # Agent, scorer, HITL nodes
├── payments/        # Slip2Go, PromptPay QR
├── runtime.py       # Shared Postgres pool for CLI/demo
├── tools/           # Catalog, sales tools
├── cli.py           # Interactive chat
└── main.py          # HITL demo
```

## Why `heuristics.py` exists

Gemini occasionally misbehaves in ways a prompt alone did not fix. Each helper
in [`ai_sales/heuristics.py`](ai_sales/heuristics.py) patches one failure we saw
in real LINE chats:

| Symptom | Helper | What it does |
| --- | --- | --- |
| Model says "ขอค้นหาสักครู่นะคะ" but emits no tool call, so the turn ends on filler | `_looks_like_search_filler` | Detected in `sales_agent_node`, which re-invokes once and forces a tool call |
| Vague questions ("มีอะไรขายบ้าง", "มีรุ่นไหนแนะนำ", "งบ 500 ซื้ออะไรได้") get "ไม่เข้าใจ" | `_should_auto_browse_catalog`, `_looks_like_confused_reply`, `_extract_budget_ceiling` | Route straight to `list_products`, with the budget as `max_price` |
| Meta words ("รุ่น", "แนะนำ", "หน่อย") sent to vector search return nothing | `_normalize_search_query`, `_is_broad_browse_query` | Rewrite to a concrete keyword and fall back to the catalog |
| Customer shows buying intent and the model restarts the conversation | `_CLOSING_INTENT` | Adds a one-turn hint to ask which offered item they want |

These are plain marker lists and regexes with no I/O. They are deliberately
blunt and Thai-specific. When a new quirk shows up, add a marker or helper
there, with a test in `tests/`, instead of growing the nodes or tools modules.
If the model improves and a helper stops firing, delete it.

## License

MIT
