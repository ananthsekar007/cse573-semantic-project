from __future__ import annotations

import json
import logging
import multiprocessing
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "backend" / "data"
EMBEDDING_DIR = DATA_ROOT / "embeddings"
PROCESSED_DIR = DATA_ROOT / "processed"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"

RetrievalMode = Literal["semantic-faiss", "tfidf-fallback"]


class ServiceError(RuntimeError):
    pass


class ConfigurationError(ServiceError):
    pass


def _summarize_text(value: str, limit: int = 220) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}... [truncated {len(compact) - limit} chars]"


def _parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class AppSettings:
    llm_api_key: str
    llm_model: str
    llm_base_url: str
    llm_timeout_seconds: int
    embedding_timeout_seconds: int
    query_retrieval_mode: Literal["auto", "semantic", "tfidf"]
    preload_assets: bool


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    log.info("Loading application settings from environment and .env files.")
    file_values: dict[str, str] = {}
    for candidate in (
        REPO_ROOT / ".env",
        REPO_ROOT / "backend" / "app" / ".env",
    ):
        file_values.update(_load_env_file(candidate))

    def env(name: str, default: str) -> str:
        return os.getenv(name, file_values.get(name, default))

    query_retrieval_mode = env("QUERY_RETRIEVAL_MODE", "auto").strip().lower()
    if query_retrieval_mode not in {"auto", "semantic", "tfidf"}:
        raise ConfigurationError(
            "QUERY_RETRIEVAL_MODE must be one of: auto, semantic, tfidf"
        )

    timeout_raw = env("LLM_TIMEOUT_SECONDS", "60")
    try:
        llm_timeout_seconds = int(timeout_raw)
    except ValueError as exc:
        raise ConfigurationError("LLM_TIMEOUT_SECONDS must be an integer") from exc

    embedding_timeout_raw = env("EMBEDDING_TIMEOUT_SECONDS", "20")
    try:
        embedding_timeout_seconds = int(embedding_timeout_raw)
    except ValueError as exc:
        raise ConfigurationError("EMBEDDING_TIMEOUT_SECONDS must be an integer") from exc

    settings = AppSettings(
        llm_api_key=env("LLM_API_KEY", "your_llm_api_key_here"),
        llm_model=env("LLM_MODEL", "llama-3.1-8b-instant"),
        llm_base_url=env("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_timeout_seconds=max(llm_timeout_seconds, 1),
        embedding_timeout_seconds=max(embedding_timeout_seconds, 1),
        query_retrieval_mode=query_retrieval_mode,
        preload_assets=_parse_bool(env("PRELOAD_PATENT_ASSETS", "1"), True),
    )
    log.info(
        "Settings loaded. query_retrieval_mode=%s preload_assets=%s llm_configured=%s llm_model=%s embedding_timeout_seconds=%s",
        settings.query_retrieval_mode,
        settings.preload_assets,
        bool(settings.llm_api_key and settings.llm_api_key != "your_llm_api_key_here"),
        settings.llm_model,
        settings.embedding_timeout_seconds,
    )
    return settings


def _local_sbert_cache_exists() -> bool:
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-mpnet-base-v2",
        Path.home() / ".cache" / "torch" / "sentence_transformers" / "sentence-transformers_all-mpnet-base-v2",
    ]
    exists = any(path.exists() for path in candidates)
    log.info("Checked local SBERT cache. exists=%s", exists)
    return exists


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer | None:
    settings = get_settings()
    if settings.query_retrieval_mode == "tfidf":
        log.info("Skipping SBERT load because retrieval mode is forced to tfidf.")
        return None
    if not _local_sbert_cache_exists():
        if settings.query_retrieval_mode == "semantic":
            raise ConfigurationError(
                "QUERY_RETRIEVAL_MODE=semantic but the local SBERT model cache is unavailable."
            )
        log.warning("SBERT cache missing; semantic retrieval unavailable, falling back if allowed.")
        return None
    try:
        log.info("Loading SBERT model from local cache: %s", SBERT_MODEL)
        model = SentenceTransformer(SBERT_MODEL, device="cpu", local_files_only=True)
        log.info("SBERT model loaded successfully.")
        return model
    except Exception as exc:
        if settings.query_retrieval_mode == "semantic":
            raise ConfigurationError(
                "Semantic retrieval is required but the local SBERT query model failed to load."
            ) from exc
        log.warning("Semantic query model unavailable; falling back to TF-IDF retrieval.")
        return None


@lru_cache(maxsize=1)
def get_http_session() -> requests.Session:
    log.info("Creating shared HTTP session for LLM requests.")
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


@lru_cache(maxsize=1)
def load_doc_index() -> faiss.Index:
    path = EMBEDDING_DIR / "faiss_doc.index"
    if not path.exists():
        raise ConfigurationError(f"Missing FAISS doc index: {path}")
    log.info("Loading document FAISS index from %s", path)
    return faiss.read_index(str(path))


@lru_cache(maxsize=1)
def load_chunk_index() -> faiss.Index:
    path = EMBEDDING_DIR / "faiss_chunk.index"
    if not path.exists():
        raise ConfigurationError(f"Missing FAISS chunk index: {path}")
    log.info("Loading chunk FAISS index from %s", path)
    return faiss.read_index(str(path))


@lru_cache(maxsize=1)
def load_doc_metadata() -> list[dict]:
    path = EMBEDDING_DIR / "doc_metadata.json"
    log.info("Loading document metadata from %s", path)
    return _load_json(path)


@lru_cache(maxsize=1)
def load_chunk_metadata() -> list[dict]:
    path = EMBEDDING_DIR / "chunk_metadata.json"
    log.info("Loading chunk metadata from %s", path)
    return _load_json(path)


@lru_cache(maxsize=1)
def load_doc_metadata_map() -> dict[str, dict]:
    return {row["patent_id"]: row for row in load_doc_metadata()}


@lru_cache(maxsize=1)
def load_doc_corpus() -> tuple[TfidfVectorizer, np.ndarray]:
    log.info("Building TF-IDF document corpus.")
    texts = []
    for row in load_doc_metadata():
        path = PROCESSED_DIR / "doc_txt" / f"{row['patent_id']}.txt"
        texts.append(path.read_text(encoding="utf-8").strip() if path.exists() else "")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    log.info("TF-IDF document corpus built. documents=%s vocabulary=%s", len(texts), len(vectorizer.vocabulary_))
    return vectorizer, matrix


@lru_cache(maxsize=1)
def load_chunk_corpus() -> tuple[TfidfVectorizer, np.ndarray]:
    log.info("Building TF-IDF chunk corpus.")
    texts = [row["text"] for row in load_chunk_metadata()]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=16000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    log.info("TF-IDF chunk corpus built. chunks=%s vocabulary=%s", len(texts), len(vectorizer.vocabulary_))
    return vectorizer, matrix


def preload_runtime_assets() -> None:
    log.info("Preloading runtime assets.")
    load_doc_metadata()
    load_chunk_metadata()
    load_doc_index()
    load_chunk_index()
    model = load_embedding_model()
    if get_settings().query_retrieval_mode == "semantic":
        log.info("Preload complete with semantic-only mode.")
        return
    if model is None:
        # Warm the fallback retrieval corpora so first novelty/chat requests
        # do not spend their whole latency budget fitting TF-IDF indexes.
        load_doc_corpus()
        load_chunk_corpus()
    log.info("Runtime asset preload finished.")


def get_runtime_status() -> dict:
    log.info("Computing runtime status.")
    settings = get_settings()
    semantic_model = load_embedding_model()
    return {
        "configured_retrieval_mode": settings.query_retrieval_mode,
        "semantic_query_model_loaded": semantic_model is not None,
        "llm_configured": bool(
            settings.llm_api_key and settings.llm_api_key != "your_llm_api_key_here"
        ),
        "doc_index_vectors": load_doc_index().ntotal,
        "chunk_index_vectors": load_chunk_index().ntotal,
    }


def embed_query(text: str) -> np.ndarray:
    log.info("Embedding query text. chars=%s preview=%s", len(text), _summarize_text(text))
    model = load_embedding_model()
    if model is None:
        raise ConfigurationError("Semantic query embedding is unavailable in this environment.")
    vector = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    log.info("Query embedding completed. shape=%s", vector.shape)
    return vector


def _encode_query_worker(text: str, output_queue) -> None:
    try:
        model = SentenceTransformer(SBERT_MODEL, device="cpu", local_files_only=True)
        vector = model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        output_queue.put(("ok", vector))
    except Exception as exc:
        output_queue.put(("error", repr(exc)))


def embed_query_with_timeout(text: str) -> np.ndarray:
    settings = get_settings()
    timeout_seconds = settings.embedding_timeout_seconds
    log.info("Embedding query with subprocess timeout. timeout_seconds=%s", timeout_seconds)

    ctx = multiprocessing.get_context("spawn")
    output_queue = ctx.Queue()
    process = ctx.Process(target=_encode_query_worker, args=(text, output_queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        log.error("Embedding query timed out after %ss; terminating subprocess.", timeout_seconds)
        process.terminate()
        process.join(5)
        raise TimeoutError(f"Semantic query embedding timed out after {timeout_seconds} seconds.")

    if output_queue.empty():
        exit_code = process.exitcode
        raise RuntimeError(f"Embedding subprocess exited without a result. exit_code={exit_code}")

    status, payload = output_queue.get()
    if status == "error":
        raise RuntimeError(f"Embedding subprocess failed: {payload}")

    vector = payload
    log.info("Embedding subprocess completed successfully. shape=%s", vector.shape)
    return vector


def build_candidate_text(title: str, abstract: str, claim_text: str) -> str:
    parts = [part.strip() for part in [title, abstract, claim_text] if part and part.strip()]
    return ". ".join(parts)


def _semantic_or_tfidf_retrieval() -> RetrievalMode:
    mode = "semantic-faiss" if load_embedding_model() is not None else "tfidf-fallback"
    log.info("Selected retrieval mode: %s", mode)
    return mode


def score_novelty(title: str, abstract: str, claim_text: str, top_k: int = 5) -> dict:
    query_text = build_candidate_text(title, abstract, claim_text)
    log.info(
        "Starting novelty scoring. top_k=%s title_len=%s abstract_len=%s claim_len=%s query_preview=%s",
        top_k,
        len(title or ""),
        len(abstract or ""),
        len(claim_text or ""),
        _summarize_text(query_text),
    )
    if not query_text:
        raise ValueError("Provide at least one of title, abstract, or claim text.")

    metadata = load_doc_metadata()
    matches = []
    retrieval_mode = _semantic_or_tfidf_retrieval()

    if retrieval_mode == "semantic-faiss":
        try:
            log.info("Novelty scoring using semantic FAISS retrieval.")
            query_vec = embed_query_with_timeout(query_text)
            index = load_doc_index()
            search_k = min(max(top_k, 1), len(metadata))
            log.info("Running FAISS search for novelty scoring. search_k=%s corpus_size=%s", search_k, len(metadata))
            distances, neighbors = index.search(query_vec, search_k)
            for rank, (neighbor_idx, similarity) in enumerate(zip(neighbors[0], distances[0]), start=1):
                item = metadata[int(neighbor_idx)]
                matches.append(
                    {
                        "rank": rank,
                        "patent_id": item.get("patent_id", f"row_{neighbor_idx}"),
                        "title": item.get("title", ""),
                        "domain": item.get("domain", "unknown"),
                        "similarity": float(similarity),
                        "novelty_score": float(1.0 - similarity),
                    }
                )
        except Exception as exc:
            log.exception(
                "Semantic novelty retrieval failed; falling back to TF-IDF for this request: %s",
                exc,
            )
            retrieval_mode = "tfidf-fallback"

    if retrieval_mode == "tfidf-fallback":
        log.info("Novelty scoring using TF-IDF fallback retrieval.")
        vectorizer, matrix = load_doc_corpus()
        query_matrix = vectorizer.transform([query_text])
        similarities = cosine_similarity(query_matrix, matrix)[0]
        ranked = np.argsort(similarities)[::-1][: max(top_k, 1)]
        for rank, neighbor_idx in enumerate(ranked, start=1):
            item = metadata[int(neighbor_idx)]
            similarity = float(similarities[int(neighbor_idx)])
            matches.append(
                {
                    "rank": rank,
                    "patent_id": item.get("patent_id", f"row_{neighbor_idx}"),
                    "title": item.get("title", ""),
                    "domain": item.get("domain", "unknown"),
                    "similarity": similarity,
                    "novelty_score": float(1.0 - similarity),
                }
            )

    top_similarity = matches[0]["similarity"] if matches else None
    novelty_score = float(1.0 - top_similarity) if top_similarity is not None else None
    interpretation = (
        "High similarity to prior art in this corpus."
        if top_similarity is not None and top_similarity >= 0.75
        else "Moderate similarity to prior art in this corpus."
        if top_similarity is not None and top_similarity >= 0.55
        else "Relatively novel with respect to this corpus."
    )

    result = {
        "query_text": query_text,
        "retrieval_mode": retrieval_mode,
        "novelty_score": novelty_score,
        "top_similarity": top_similarity,
        "interpretation": interpretation,
        "matches": matches,
    }
    log.info(
        "Novelty scoring finished. retrieval_mode=%s matches=%s top_similarity=%s novelty_score=%s",
        retrieval_mode,
        len(matches),
        top_similarity,
        novelty_score,
    )
    return result


def retrieve_rag_chunks(question: str, top_k: int = 6) -> tuple[list[dict], RetrievalMode]:
    log.info(
        "Starting RAG retrieval. top_k=%s question_len=%s preview=%s",
        top_k,
        len(question or ""),
        _summarize_text(question),
    )
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    chunk_metadata = load_chunk_metadata()
    doc_meta_map = load_doc_metadata_map()
    retrieval_mode = _semantic_or_tfidf_retrieval()

    if retrieval_mode == "semantic-faiss":
        try:
            log.info("RAG retrieval using semantic FAISS.")
            query_vec = embed_query_with_timeout(question)
            index = load_chunk_index()
            search_k = min(max(top_k, 1), len(chunk_metadata))
            log.info("Running FAISS search for RAG. search_k=%s chunk_count=%s", search_k, len(chunk_metadata))
            distances, neighbors = index.search(query_vec, search_k)
            ranking = [(int(chunk_idx), float(similarity)) for chunk_idx, similarity in zip(neighbors[0], distances[0])]
        except Exception as exc:
            log.exception(
                "Semantic RAG retrieval failed; falling back to TF-IDF for this request: %s",
                exc,
            )
            retrieval_mode = "tfidf-fallback"

    if retrieval_mode == "tfidf-fallback":
        log.info("RAG retrieval using TF-IDF fallback.")
        vectorizer, matrix = load_chunk_corpus()
        query_matrix = vectorizer.transform([question])
        similarities = cosine_similarity(query_matrix, matrix)[0]
        ranked = np.argsort(similarities)[::-1][: max(top_k, 1)]
        ranking = [(int(chunk_idx), float(similarities[int(chunk_idx)])) for chunk_idx in ranked]

    results = []
    for rank, (chunk_idx, similarity) in enumerate(ranking, start=1):
        item = chunk_metadata[int(chunk_idx)]
        patent = doc_meta_map.get(item["patent_id"], {})
        results.append(
            {
                "rank": rank,
                "patent_id": item["patent_id"],
                "title": patent.get("title", ""),
                "domain": patent.get("domain", "unknown"),
                "chunk_index": item["chunk_index"],
                "similarity": float(similarity),
                "text": item["text"],
            }
        )
    log.info("RAG retrieval finished. retrieval_mode=%s chunks=%s", retrieval_mode, len(results))
    return results, retrieval_mode


def build_local_rag_answer(question: str, chunks: list[dict]) -> str:
    log.info(
        "Building local retrieval-only RAG answer. question_preview=%s chunks=%s",
        _summarize_text(question),
        len(chunks),
    )
    if not chunks:
        return "No supporting passages were retrieved from the patent corpus."

    domains = Counter(chunk["domain"] for chunk in chunks)
    patents = []
    seen = set()
    for chunk in chunks:
        if chunk["patent_id"] in seen:
            continue
        seen.add(chunk["patent_id"])
        patents.append(f"{chunk['patent_id']} ({chunk['title']})")
        if len(patents) == 3:
            break

    dominant_domains = ", ".join(domain for domain, _count in domains.most_common(3))
    return (
        f"LLM generation is unavailable, so this is a retrieval-only response. "
        f"The question appears most related to these domains: {dominant_domains}. "
        f"The strongest supporting patents retrieved were {', '.join(patents)}."
    )


def call_llm_chat(question: str, chunks: list[dict]) -> tuple[str, str]:
    settings = get_settings()
    if not settings.llm_api_key or settings.llm_api_key == "your_llm_api_key_here":
        log.warning("LLM API key missing; returning retrieval-only response.")
        return build_local_rag_answer(question, chunks), "retrieval-only"

    context = "\n\n".join(
        [
            f"[Patent {chunk['patent_id']} | {chunk['title']} | score={chunk['similarity']:.3f}]\n{chunk['text']}"
            for chunk in chunks
        ]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a patent analysis assistant. Answer only from the provided retrieval context. "
                "Be concise, mention uncertainty when needed, and cite patent ids explicitly."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nRetrieved context:\n{context}",
        },
    ]

    try:
        log.info(
            "Calling LLM chat API. base_url=%s model=%s chunks=%s timeout=%ss",
            settings.llm_base_url,
            settings.llm_model,
            len(chunks),
            settings.llm_timeout_seconds,
        )
        response = get_http_session().post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_model,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
        log.info("LLM chat call succeeded.")
        return answer, "llm"
    except Exception as exc:
        log.warning("LLM call failed; returning retrieval-only response: %s", exc)
        return build_local_rag_answer(question, chunks), "retrieval-only"


def run_rag_chat(question: str, top_k: int = 6) -> dict:
    log.info("Running RAG chat flow.")
    chunks, retrieval_mode = retrieve_rag_chunks(question, top_k=top_k)
    answer, mode = call_llm_chat(question, chunks)

    supporting_patents = []
    seen = set()
    for chunk in chunks:
        if chunk["patent_id"] in seen:
            continue
        seen.add(chunk["patent_id"])
        supporting_patents.append(
            {
                "patent_id": chunk["patent_id"],
                "title": chunk["title"],
                "domain": chunk["domain"],
            }
        )

    result = {
        "question": question,
        "answer": answer,
        "mode": mode,
        "retrieval_mode": retrieval_mode,
        "retrieved_chunks": chunks,
        "supporting_patents": supporting_patents,
    }
    log.info(
        "RAG chat flow finished. mode=%s retrieval_mode=%s retrieved_chunks=%s supporting_patents=%s",
        mode,
        retrieval_mode,
        len(chunks),
        len(supporting_patents),
    )
    return result
