"""
FastAPI server exposing the AI Sales agent as the "brain" behind Next.js.

The Next.js app (front desk + manager) sends JSON commands here; this service
runs the LangGraph reasoning, performs vector search in Pinecone, and returns
the agent's reply plus lead-scoring / discount-approval state.

Endpoints
---------
GET  /health                 Liveness probe.
POST /chat                   Send a customer message, get the agent reply + state.
POST /approval               Manager approves/rejects a pending discount (HITL resume).
GET  /state/{thread_id}      Inspect the current state of a conversation thread.
POST /payments/verify-slip   Verify a payment slip (Base64 image) via Slip2Go.
POST /payments/promptpay-qr  Generate a PromptPay QR for receiving payment.
GET  /payments/slip/{ref}    Fetch a previously verified slip (anti-reuse check).

Run with:
    poetry run python -m ai_sales serve
    # or
    poetry run uvicorn ai_sales.api.server:app --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# AGENTS.md: Always run load_dotenv() at the top so credentials are available.
load_dotenv()

import base64
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai_sales.api import service
from ai_sales.api import tasks
from ai_sales.api.retry import is_retryable_error
from ai_sales.knowledge import indexer
from ai_sales.payments import qr as payment_qr_module
from ai_sales.payments import slip2go

logger = logging.getLogger(__name__)


def _is_production() -> bool:
    return os.getenv("ENV", "").strip().lower() in ("production", "prod")


def _validate_production_env() -> None:
    if not _is_production():
        return
    required = [
        "BRAIN_API_KEY",
        "INTERNAL_API_KEY",
        "GEMINI_API_KEY",
        "DASHBOARD_URL",
    ]
    missing = [key for key in required if not (os.getenv(key) or "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variables for production: "
            + ", ".join(missing)
        )


def _internal_error(exc: Exception) -> HTTPException:
    logger.exception("Unhandled brain error")
    if _is_production():
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ChatDelivery(BaseModel):
    """Optional outbound delivery hints for async /chat jobs."""

    line_push_target: str | None = Field(
        None,
        description="LINE userId / groupId / roomId to push the final reply to",
    )
    display_name: str | None = Field(
        None,
        description="Customer display name (stored on orders / metadata)",
    )
    attach_payment_qr: dict | None = Field(
        None,
        description="Optional QR payload to push with the reply (e.g. partial balance)",
    )


class ChatRequest(BaseModel):
    """A single customer message bound to a conversation thread."""

    message: str = Field(..., min_length=1, description="Customer message text")
    thread_id: str = Field(
        ...,
        min_length=1,
        description="Conversation id (one thread per conversation)",
    )
    payment_context: dict | None = Field(
        None,
        description="Optional payment reconciliation context (overpay amounts)",
    )
    async_mode: bool = Field(
        False,
        description=(
            "When true, return immediately with status=accepted and run LangGraph "
            "in a background task (LINE push + dashboard callback when done)."
        ),
    )
    delivery: ChatDelivery | None = Field(
        None,
        description="Outbound delivery options (required for LINE async flow)",
    )


class ApprovalRequest(BaseModel):
    """A manager's decision on a pending discount approval."""

    thread_id: str = Field(..., min_length=1)
    approved: bool = Field(..., description="True to approve, False to reject")


class ChatAcceptedResponse(BaseModel):
    """Immediate acknowledgement for async /chat (LangGraph runs in background)."""

    thread_id: str
    status: str = "accepted"
    reply: str = "รับทราบ!"
    async_mode: bool = True


class AgentResponse(BaseModel):
    """The agent reply plus a snapshot of the conversation state."""

    thread_id: str
    reply: str
    lead_score: int = 0
    pipeline_stage: str = "new"
    requires_approval: bool = False
    pending_discount_approval: dict | None = None
    next_nodes: list[str] = []
    resumed: bool | None = None
    shipping_info: dict | None = None
    order_ready: bool = False
    payment_qr: dict | None = None
    pending_overpay: dict | None = None
    overpay_resolution: str | None = None
    overpay_credit_amount: float = 0
    awaiting_refund_approval: bool = False


class VerifySlipRequest(BaseModel):
    """A payment slip to verify (Base64 image) with optional check conditions."""

    image_base64: str = Field(
        ...,
        min_length=1,
        description="Slip image as Base64 (may include the data:image/...;base64, prefix)",
    )
    check_condition: dict | None = Field(
        None,
        description="Optional Slip2Go checkCondition (checkDuplicate/checkReceiver/checkAmount/checkDate)",
    )


class PromptPayQRRequest(BaseModel):
    """Parameters to generate a PromptPay QR for receiving payment."""

    prompt_pay_code: str = Field(..., min_length=1, description="PromptPay id")
    prompt_pay_type: Literal["phone_number", "citizen_id", "e_wallet"] = (
        "phone_number"
    )
    account_name: str | None = Field(None, description="Receiver display name")
    amount: float | None = Field(None, description="Amount in THB (optional)")


class KnowledgeIndexRequest(BaseModel):
    """Index a Knowledge Document's plain text into the vector DB."""

    document_id: str = Field(..., min_length=1)
    title: str = Field("", description="Document title (stored as chunk metadata)")
    text: str = Field("", description="Plain text content to chunk + embed")


class KnowledgeIndexFileRequest(BaseModel):
    """Index an uploaded file (PDF/TXT/CSV) into the vector DB."""

    document_id: str = Field(..., min_length=1)
    title: str = Field("", description="Document title")
    filename: str = Field(..., min_length=1, description="Original filename (for type detection)")
    content_base64: str = Field(..., min_length=1, description="File bytes as Base64")


class KnowledgeDeleteRequest(BaseModel):
    """Delete all vectors belonging to a Knowledge Document."""

    document_id: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Auth for server-to-server (dashboard → brain) endpoints
# ---------------------------------------------------------------------------
def require_brain_key(x_brain_key: str | None = Header(default=None)) -> None:
    """Guard brain endpoints with a shared secret.

    In production (``ENV=production``), ``BRAIN_API_KEY`` is mandatory.
    In local dev, an unset key skips the check for convenience.
    """
    expected = (os.getenv("BRAIN_API_KEY") or "").strip()
    if not expected:
        if _is_production():
            raise HTTPException(
                status_code=500,
                detail="BRAIN_API_KEY is not configured",
            )
        return
    if x_brain_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing brain API key")


BRAIN_AUTH = [Depends(require_brain_key)]


# ---------------------------------------------------------------------------
# App lifecycle: build the graph once at startup, close it on shutdown.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_production_env()
    # Build the graph eagerly so the first request is fast and startup fails
    # loudly if the LLM/vector store configuration is wrong.
    service.get_graph()
    try:
        yield
    finally:
        service.close()


app = FastAPI(
    title="AI Sales Brain",
    description="LangGraph sales agent exposed over HTTP for the Next.js dashboard.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: allow the Next.js origin(s). In production Next.js calls this from the
# server (BFF proxy), but allowing the dev origin keeps local testing easy.
_allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Simple liveness probe used by Next.js / orchestration."""
    return {"status": "ok", "service": "ai-sales-brain"}


@app.get("/health/ready")
def health_ready() -> dict:
    """Readiness: graph compiled and required credentials present."""
    checks: dict[str, str] = {}
    try:
        service.get_graph()
        checks["graph"] = "ok"
    except Exception as exc:
        checks["graph"] = f"error: {exc}"

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        checks["llm"] = "missing GEMINI_API_KEY"
    else:
        checks["llm"] = "ok"

    if _is_production():
        for key in ("BRAIN_API_KEY", "INTERNAL_API_KEY", "DASHBOARD_URL"):
            checks[key.lower()] = "ok" if (os.getenv(key) or "").strip() else "missing"

    ready = all(v == "ok" for v in checks.values())
    if not ready:
        raise HTTPException(status_code=503, detail={"ready": False, "checks": checks})
    return {"ready": True, "checks": checks}


class AsyncChatRequest(BaseModel):
    """Enqueue payload for LINE / fire-and-forget integrations."""

    message: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    payment_context: dict | None = None
    delivery: ChatDelivery | None = None


def _spawn_chat_worker(
    thread_id: str,
    message: str,
    payment_context: dict | None,
    line_push_target: str | None,
    display_name: str | None,
    attach_payment_qr: dict | None = None,
) -> None:
    """Run LangGraph on a dedicated thread so HTTP workers never block."""
    worker = threading.Thread(
        target=tasks.process_chat_async,
        args=(
            thread_id,
            message,
            payment_context,
            line_push_target,
            display_name,
            attach_payment_qr,
        ),
        daemon=True,
        name=f"chat-{thread_id[:24]}",
    )
    worker.start()


def _accept_async_chat(req: ChatRequest | AsyncChatRequest) -> JSONResponse:
    delivery = req.delivery or ChatDelivery()
    _spawn_chat_worker(
        req.thread_id,
        req.message,
        req.payment_context,
        delivery.line_push_target,
        delivery.display_name,
        delivery.attach_payment_qr,
    )
    return JSONResponse(
        status_code=202,
        content=ChatAcceptedResponse(thread_id=req.thread_id).model_dump(),
    )


@app.post("/chat/async", response_model=None, status_code=202, dependencies=BRAIN_AUTH)
async def chat_async(req: AsyncChatRequest) -> JSONResponse:
    """Ack immediately; LangGraph + LINE push run in a background thread."""
    return _accept_async_chat(req)


@app.post("/chat", response_model=None, dependencies=BRAIN_AUTH)
async def chat(req: ChatRequest):
    """Process a customer message.

    * ``async_mode=false`` (default): blocking — returns the full agent reply.
    * ``async_mode=true``: returns ``202 accepted`` immediately; LangGraph runs
      in a background task, then pushes LINE (if configured) and calls the
      dashboard callback to persist messages/orders.
    """
    if req.async_mode:
        return _accept_async_chat(req)

    try:
        result = await asyncio.to_thread(
            service.send_message,
            req.thread_id,
            req.message,
            req.payment_context,
        )
    except Exception as exc:
        if is_retryable_error(exc):
            raise HTTPException(
                status_code=503,
                detail="บริการ AI ไม่พร้อมชั่วคราว กรุณาลองใหม่อีกครั้ง",
            ) from exc
        raise _internal_error(exc)
    return AgentResponse(**result)


@app.post("/approval", response_model=AgentResponse, dependencies=BRAIN_AUTH)
def approval(req: ApprovalRequest) -> AgentResponse:
    """Resume a paused conversation with the manager's discount decision."""
    try:
        result = service.resume_with_approval(req.thread_id, req.approved)
    except Exception as exc:
        if is_retryable_error(exc):
            raise HTTPException(
                status_code=503,
                detail="บริการ AI ไม่พร้อมชั่วคราว กรุณาลองใหม่อีกครั้ง",
            ) from exc
        raise _internal_error(exc)
    return AgentResponse(**result)


@app.get("/state/{thread_id}", response_model=AgentResponse, dependencies=BRAIN_AUTH)
def state(thread_id: str) -> AgentResponse:
    """Return the current state snapshot for a conversation thread."""
    try:
        result = service.get_state(thread_id)
    except Exception as exc:
        raise _internal_error(exc)
    # No reply field for a pure state read.
    result.setdefault("reply", "")
    return AgentResponse(**result)


# ---------------------------------------------------------------------------
# Payments (Slip2Go): slip verification + PromptPay QR generation
# ---------------------------------------------------------------------------
@app.post("/payments/verify-slip", dependencies=BRAIN_AUTH)
def verify_slip(req: VerifySlipRequest) -> dict:
    """Verify a bank-transfer slip from a Base64 image via Slip2Go."""
    try:
        return slip2go.verify_slip_base64(req.image_base64, req.check_condition)
    except slip2go.Slip2GoError as exc:
        raise HTTPException(
            status_code=exc.status or 502,
            detail={"message": str(exc), "provider": exc.payload},
        ) from exc


@app.post("/payments/promptpay-qr", dependencies=BRAIN_AUTH)
def promptpay_qr(req: PromptPayQRRequest) -> dict:
    """Generate a PromptPay QR code for receiving payment via Slip2Go."""
    try:
        return slip2go.generate_promptpay_qr(
            prompt_pay_code=req.prompt_pay_code,
            prompt_pay_type=req.prompt_pay_type,
            account_name=req.account_name or "",
            amount=req.amount,
        )
    except slip2go.Slip2GoError as exc:
        raise HTTPException(
            status_code=exc.status or 502,
            detail={"message": str(exc), "provider": exc.payload},
        ) from exc


class PartialQrRequest(BaseModel):
    """Generate a dynamic QR PNG for an exact partial/missing amount."""

    amount: float = Field(..., gt=0)
    items: str = ""


@app.post("/payments/partial-qr", dependencies=BRAIN_AUTH)
def partial_qr(req: PartialQrRequest) -> dict:
    """Return a PaymentQr-shaped dict with PNG base64 for a specific amount."""
    try:
        return payment_qr_module.create_promptpay_qr(
            req.amount, req.items, partial=True
        )
    except slip2go.Slip2GoError as exc:
        raise HTTPException(
            status_code=exc.status or 502,
            detail={"message": str(exc), "provider": exc.payload},
        ) from exc


@app.get("/payments/slip/{reference_id}", dependencies=BRAIN_AUTH)
def get_slip(reference_id: str) -> dict:
    """Fetch a previously verified slip by referenceId (anti-reuse check).

    Does not consume verification quota — use it to confirm whether a slip
    reference was already seen before accepting a customer's payment.
    """
    try:
        return slip2go.get_slip_by_reference(reference_id)
    except slip2go.Slip2GoError as exc:
        raise HTTPException(
            status_code=exc.status or 502,
            detail={"message": str(exc), "provider": exc.payload},
        ) from exc


# ---------------------------------------------------------------------------
# Knowledge Base (unstructured docs → Pinecone for RAG)
# ---------------------------------------------------------------------------
@app.post("/knowledge/index", dependencies=BRAIN_AUTH)
def knowledge_index(req: KnowledgeIndexRequest) -> dict:
    """Chunk + embed a document's plain text and upsert it into the vector DB."""
    try:
        chunk_count = indexer.index_document(req.document_id, req.title, req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"document_id": req.document_id, "chunk_count": chunk_count}


@app.post("/knowledge/index-file", dependencies=BRAIN_AUTH)
def knowledge_index_file(req: KnowledgeIndexFileRequest) -> dict:
    """Extract text from an uploaded file, then chunk + embed + upsert it."""
    try:
        raw = base64.b64decode(req.content_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Base64: {exc}") from exc

    try:
        text = indexer.extract_text(raw, req.filename)
        chunk_count = indexer.index_document(req.document_id, req.title, text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "document_id": req.document_id,
        "chunk_count": chunk_count,
    }


@app.post("/knowledge/delete", dependencies=BRAIN_AUTH)
def knowledge_delete(req: KnowledgeDeleteRequest) -> dict:
    """Delete all vectors belonging to a Knowledge Document."""
    try:
        deleted = indexer.delete_document(req.document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"document_id": req.document_id, "deleted": deleted}
