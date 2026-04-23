from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .data import load_cluster_dashboard


app = FastAPI(title="Patent Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/api/dashboard/clusters")
def cluster_dashboard() -> dict:
    return load_cluster_dashboard()
