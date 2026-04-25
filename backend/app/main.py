from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .data import load_cluster_dashboard
from .services import ConfigurationError
from .services import get_runtime_status
from .services import get_settings
from .services import preload_runtime_assets
from .services import run_rag_chat
from .services import score_novelty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


def _summarize_text(value: str, limit: int = 300) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}... [truncated {len(compact) - limit} chars]"


def _summarize_json_body(raw_body: bytes) -> str:
    if not raw_body:
        return "<empty>"

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError:
        return _summarize_text(raw_body.decode("utf-8", errors="replace"))

    if isinstance(parsed, dict):
        summary = {}
        for key, value in parsed.items():
            if isinstance(value, str):
                summary[key] = _summarize_text(value)
            else:
                summary[key] = value
        return json.dumps(summary, ensure_ascii=True)
    return _summarize_text(json.dumps(parsed, ensure_ascii=True))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Application startup beginning.")
    settings = get_settings()
    log.info(
        "Loaded settings. preload_assets=%s query_retrieval_mode=%s llm_model=%s llm_base_url=%s",
        settings.preload_assets,
        settings.query_retrieval_mode,
        settings.llm_model,
        settings.llm_base_url,
    )
    if settings.preload_assets:
        log.info("Preloading runtime assets during startup.")
        preload_runtime_assets()
        log.info("Runtime asset preload complete.")
    yield
    log.info("Application shutdown complete.")


app = FastAPI(title="Patent Dashboard API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
    started_at = time.perf_counter()
    raw_body = await request.body()
    body_summary = _summarize_json_body(raw_body)

    async def receive():
        return {"type": "http.request", "body": raw_body, "more_body": False}

    request._receive = receive
    log.info(
        "[%s] Incoming request: %s %s content_type=%s content_length=%s body=%s",
        request_id,
        request.method,
        request.url.path,
        request.headers.get("content-type", "<missing>"),
        request.headers.get("content-length", "<missing>"),
        body_summary,
    )

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        log.exception(
            "[%s] Request crashed before response: %s %s after %.1fms",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    log.info(
        "[%s] Request completed: %s %s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    log.warning(
        "Validation failed for %s %s: errors=%s",
        request.method,
        request.url.path,
        errors,
    )
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/api/health")
def healthcheck() -> dict:
    try:
        log.info("Healthcheck requested.")
        return {"status": "ok", "runtime": get_runtime_status()}
    except ConfigurationError as exc:
        log.warning("Healthcheck degraded due to configuration error: %s", exc)
        return {"status": "degraded", "detail": str(exc)}


@app.get("/api/dashboard/clusters")
def cluster_dashboard() -> dict:
    log.info("Cluster dashboard requested.")
    return load_cluster_dashboard()


class NoveltyRequest(BaseModel):
    title: str = ""
    abstract: str = ""
    claim_text: str = ""
    top_k: int = Field(default=5, ge=1, le=10)


class RagRequest(BaseModel):
    question: str
    top_k: int = Field(default=6, ge=1, le=10)


@app.post("/api/novelty/score")
def novelty_score(payload: NoveltyRequest) -> dict:
    log.info(
        "Novelty endpoint invoked. top_k=%s title_len=%s abstract_len=%s claim_text_len=%s",
        payload.top_k,
        len(payload.title or ""),
        len(payload.abstract or ""),
        len(payload.claim_text or ""),
    )
    try:
        result = score_novelty(
            title=payload.title,
            abstract=payload.abstract,
            claim_text=payload.claim_text,
            top_k=payload.top_k,
        )
        log.info(
            "Novelty endpoint completed. retrieval_mode=%s top_similarity=%s novelty_score=%s",
            result.get("retrieval_mode"),
            result.get("top_similarity"),
            result.get("novelty_score"),
        )
        return result
    except ConfigurationError as exc:
        log.warning("Novelty endpoint configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        log.warning("Novelty endpoint bad request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Novelty endpoint failed unexpectedly.")
        raise HTTPException(status_code=500, detail=f"Novelty scoring failed: {exc}") from exc


@app.post("/api/rag/chat")
def rag_chat(payload: RagRequest) -> dict:
    log.info(
        "RAG endpoint invoked. top_k=%s question_len=%s question_preview=%s",
        payload.top_k,
        len(payload.question or ""),
        _summarize_text(payload.question or "", limit=200),
    )
    try:
        result = run_rag_chat(question=payload.question, top_k=payload.top_k)
        log.info(
            "RAG endpoint completed. mode=%s retrieval_mode=%s chunks=%s supporting_patents=%s",
            result.get("mode"),
            result.get("retrieval_mode"),
            len(result.get("retrieved_chunks", [])),
            len(result.get("supporting_patents", [])),
        )
        return result
    except ConfigurationError as exc:
        log.warning("RAG endpoint configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        log.warning("RAG endpoint bad request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("RAG endpoint failed unexpectedly.")
        raise HTTPException(status_code=500, detail=f"RAG chat failed: {exc}") from exc
