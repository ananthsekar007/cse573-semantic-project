from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data import load_cluster_dashboard
from .services import ConfigurationError
from .services import get_runtime_status
from .services import get_settings
from .services import preload_runtime_assets
from .services import run_rag_chat
from .services import score_novelty


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    if settings.preload_assets:
        preload_runtime_assets()
    yield


app = FastAPI(title="Patent Dashboard API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def healthcheck() -> dict:
    try:
        return {"status": "ok", "runtime": get_runtime_status()}
    except ConfigurationError as exc:
        return {"status": "degraded", "detail": str(exc)}


@app.get("/api/dashboard/clusters")
def cluster_dashboard() -> dict:
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
    try:
        return score_novelty(
            title=payload.title,
            abstract=payload.abstract,
            claim_text=payload.claim_text,
            top_k=payload.top_k,
        )
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Novelty scoring failed: {exc}") from exc


@app.post("/api/rag/chat")
def rag_chat(payload: RagRequest) -> dict:
    try:
        return run_rag_chat(question=payload.question, top_k=payload.top_k)
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG chat failed: {exc}") from exc
