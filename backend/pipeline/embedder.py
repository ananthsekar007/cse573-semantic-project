"""
Patent Embedding Pipeline
==========================
Reads cleaned patent files from preprocessor.py output and produces:

  Document-level (for clustering + novelty scoring):
    embeddings/doc_embeddings.npy       — (N, 768) float32 numpy array
    embeddings/doc_metadata.json        — maps row index → patent_id, domain, title
    embeddings/faiss_doc.index          — FAISS flat index for novelty scoring

  Chunk-level (for RAG retrieval):
    embeddings/chunk_embeddings.npy     — (M, 768) float32 numpy array
    embeddings/chunk_metadata.json      — maps row index → patent_id, chunk_index, text
    embeddings/faiss_chunk.index        — FAISS flat index for RAG retrieval

Why two separate strategies:
  - Document-level: title + abstract + claim 1 as one unit → fed directly
    into HDBSCAN and UMAP as a numpy matrix. FAISS doc index is only used
    for novelty scoring (querying a new patent against the corpus).
  - Chunk-level: description + claims split into overlapping 400-token
    windows → stored in FAISS for RAG. SBERT has a 512-token hard limit so
    chunking is required to cover the full patent text.

Usage:
    python embedder.py
    python embedder.py --processed-dir ./data/processed --output-dir ./data/embeddings
    python embedder.py --batch-size 16 --chunk-size 400 --chunk-overlap 50
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SBERT_MODEL    = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM  = 768
BATCH_SIZE     = 32    # patents per SBERT batch — reduce if OOM
CHUNK_SIZE     = 280   # words per RAG chunk — safe under 384-token SBERT limit (280 × 1.3 ≈ 364 tokens)
CHUNK_OVERLAP  = 40    # word overlap between consecutive chunks
MIN_CHUNK_CHARS = 100  # discard chunks shorter than this (likely noise)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SBERT model — loaded once, reused for both pipelines
# ---------------------------------------------------------------------------

def load_model() -> SentenceTransformer:
    log.info("Loading SBERT model: %s", SBERT_MODEL)

    # Force CPU device explicitly.
    # PyTorch MPS backend (Apple Silicon) has known segfault issues on
    # Python 3.9 + macOS when used with sentence-transformers. CPU is
    # stable and fast enough for corpora under 1000 documents.
    model = SentenceTransformer(SBERT_MODEL, device="cpu")
    log.info("Model loaded on CPU. Max sequence length: %d tokens", model.max_seq_length)
    return model

# def embed_texts(model, texts, batch_size=BATCH_SIZE, desc="Embedding"):
#     outs = []
#     for idx, text in enumerate(texts):
#         print("Encoding", idx, "chars=", len(text), "preview=", repr(text[:120]))
#         vec = model.encode(
#             [text],
#             batch_size=1,
#             show_progress_bar=False,
#             normalize_embeddings=True,
#             convert_to_numpy=True,
#         )
#         print("OK", idx, vec.shape)

def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    desc: str = "Embedding",
) -> np.ndarray:
    """
    Encodes a list of texts into L2-normalised 768-dim vectors.
    normalize_embeddings=True ensures cosine similarity == dot product,
    which allows FAISS IndexFlatIP to be used as a cosine similarity index.
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # critical for cosine similarity via dot product
        convert_to_numpy=True,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Chunking — for chunk-level RAG embeddings
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits text into overlapping word-level windows.

    Uses word-level splitting rather than token-level because SBERT's
    tokenizer is internal — we approximate tokens as words (1 word ≈ 1.3
    tokens on average for English patent text). This keeps chunks safely
    under all-mpnet-base-v2's actual 384-token hard limit.

    chunk_size=280 words × 1.3 ≈ 364 tokens → safely under 384-token limit
    with headroom for longer words. Reduce to 250 if you see truncation warnings.
    """
    words  = text.split()
    chunks = []
    step   = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break

    return chunks


# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a FAISS IndexFlatIP (inner product) index.

    Because embeddings are L2-normalised, inner product == cosine similarity.
    IndexFlatIP is exact (brute force) — appropriate for corpora of 500–5000
    vectors. For 100k+ vectors, consider IndexIVFFlat with nlist=100.
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    log.info("FAISS index built: %d vectors", index.ntotal)
    return index


def save_faiss_index(index: faiss.IndexFlatIP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        faiss.write_index(index, str(path))
        log.info("FAISS index saved → %s  (%d vectors)", path, index.ntotal)
    except Exception as e:
        log.error("Failed to save FAISS index to %s: %s", path, e)
        raise


# ---------------------------------------------------------------------------
# Document-level embedding pipeline
# ---------------------------------------------------------------------------

def run_doc_pipeline(
    model:         SentenceTransformer,
    processed_dir: Path,
    output_dir:    Path,
    batch_size:    int,
) -> None:
    """
    Embeds each patent as a single document vector using:
        title + abstract_clean + first claim (clean)

    Output:
        doc_embeddings.npy    — numpy array shape (N, 768)
        doc_metadata.json     — list of {index, patent_id, domain, title}
        faiss_doc.index       — FAISS index for novelty scoring
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("━" * 60)
    log.info("DOCUMENT-LEVEL EMBEDDING PIPELINE")
    log.info("━" * 60)
    log.info("Reading from : %s", processed_dir)
    log.info("Writing to   : %s", output_dir)

    doc_txt_dir = processed_dir / "doc_txt"
    json_dir    = processed_dir / "cleaned_json"

    if not doc_txt_dir.exists():
        raise FileNotFoundError(f"doc_txt/ not found in {processed_dir}. Run preprocessor.py first.")

    doc_files = sorted(doc_txt_dir.glob("*.txt"))
    log.info("Found %d doc_txt files", len(doc_files))

    texts    = []
    metadata = []

    for path in tqdm(doc_files, desc="Loading doc texts"):
        patent_id = path.stem
        text      = path.read_text(encoding="utf-8").strip()

        if not text:
            log.warning("Empty doc_txt for %s — skipping", patent_id)
            continue

        # Pull domain and title from cleaned JSON for metadata
        json_path = json_dir / f"{patent_id}.json"
        domain    = ""
        title     = ""
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)
            domain = meta.get("domain", "")
            title  = meta.get("title", "")

        texts.append(text)
        metadata.append({
            "index":     len(texts) - 1,
            "patent_id": patent_id,
            "domain":    domain,
            "title":     title,
        })

    log.info("Embedding %d documents...", len(texts))
    embeddings = embed_texts(model, texts, batch_size=batch_size, desc="Doc embeddings")

    # Save numpy array
    npy_path = output_dir / "doc_embeddings.npy"
    np.save(npy_path, embeddings)
    log.info("Saved doc_embeddings.npy → shape %s", embeddings.shape)

    # Save metadata
    meta_path = output_dir / "doc_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Saved doc_metadata.json → %d entries", len(metadata))

    # Build and save FAISS index (for novelty scoring)
    log.info("Building FAISS doc index...")
    index      = build_faiss_index(embeddings)
    faiss_path = output_dir / "faiss_doc.index"
    save_faiss_index(index, faiss_path)

    log.info("Document pipeline complete. %d patents embedded.", len(texts))


# ---------------------------------------------------------------------------
# Chunk-level embedding pipeline
# ---------------------------------------------------------------------------

def run_chunk_pipeline(
    model:         SentenceTransformer,
    processed_dir: Path,
    output_dir:    Path,
    batch_size:    int,
    chunk_size:    int,
    chunk_overlap: int,
) -> None:
    """
    Splits each patent's rag_txt into overlapping chunks and embeds each
    chunk independently. Stores all chunks in a single FAISS index with
    metadata mapping each chunk back to its source patent.

    Source text: rag_txt/<patent_id>.txt
      Contains: title, abstract_clean, claims_clean, description_clean
      This is the full cleaned but readable text — not lemmatized.

    Output:
        chunk_embeddings.npy   — numpy array shape (M, 768) where M >> N
        chunk_metadata.json    — list of {index, patent_id, chunk_index, text}
        faiss_chunk.index      — FAISS index for RAG retrieval
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("━" * 60)
    log.info("CHUNK-LEVEL EMBEDDING PIPELINE")
    log.info("━" * 60)
    log.info("Reading from : %s", processed_dir)
    log.info("Writing to   : %s", output_dir)

    rag_txt_dir = processed_dir / "rag_txt"

    if not rag_txt_dir.exists():
        raise FileNotFoundError(f"rag_txt/ not found in {processed_dir}. Run preprocessor.py first.")

    rag_files = sorted(rag_txt_dir.glob("*.txt"))
    log.info("Found %d rag_txt files", len(rag_files))
    log.info("Chunk size: %d words | Overlap: %d words", chunk_size, chunk_overlap)

    all_chunks   = []
    all_metadata = []

    for path in tqdm(rag_files, desc="Chunking patents"):
        patent_id = path.stem
        text      = path.read_text(encoding="utf-8").strip()

        if not text:
            log.warning("Empty rag_txt for %s — skipping", patent_id)
            continue

        chunks = chunk_text(text, chunk_size, chunk_overlap)

        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "index":       len(all_chunks) - 1,
                "patent_id":   patent_id,
                "chunk_index": chunk_idx,
                "text":        chunk,          # stored for RAG context injection
            })

    log.info(
        "Total chunks: %d across %d patents (avg %.1f chunks/patent)",
        len(all_chunks),
        len(rag_files),
        len(all_chunks) / max(len(rag_files), 1),
    )

    log.info("Embedding %d chunks...", len(all_chunks))
    embeddings = embed_texts(
        model, all_chunks, batch_size=batch_size, desc="Chunk embeddings"
    )

    # Save numpy array
    npy_path = output_dir / "chunk_embeddings.npy"
    np.save(npy_path, embeddings)
    log.info("Saved chunk_embeddings.npy → shape %s", embeddings.shape)

    # Save metadata — note: chunk text is included so RAG can inject it
    # into the LLM prompt without re-reading the file at query time
    meta_path = output_dir / "chunk_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    log.info("Saved chunk_metadata.json → %d chunks", len(all_metadata))

    # Build and save FAISS index (for RAG retrieval)
    log.info("Building FAISS chunk index...")
    index      = build_faiss_index(embeddings)
    faiss_path = output_dir / "faiss_chunk.index"
    save_faiss_index(index, faiss_path)

    log.info("Chunk pipeline complete. %d chunks indexed.", len(all_chunks))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(output_dir: Path) -> None:
    log.info("")
    log.info("━" * 60)
    log.info("EMBEDDING PIPELINE SUMMARY")
    log.info("━" * 60)

    files = {
        "doc_embeddings.npy":  "Document vectors  (HDBSCAN / UMAP input)",
        "doc_metadata.json":   "Document metadata (patent_id, domain, title)",
        "faiss_doc.index":     "FAISS doc index   (novelty scoring)",
        "chunk_embeddings.npy":"Chunk vectors     (RAG input)",
        "chunk_metadata.json": "Chunk metadata    (patent_id, chunk text)",
        "faiss_chunk.index":   "FAISS chunk index (RAG retrieval)",
    }

    for filename, description in files.items():
        path = output_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            log.info("  ✓ %-30s  %.2f MB  — %s", filename, size_mb, description)
        else:
            log.warning("  ✗ %-30s  MISSING", filename)

    # Load and report shapes
    doc_npy   = output_dir / "doc_embeddings.npy"
    chunk_npy = output_dir / "chunk_embeddings.npy"

    if doc_npy.exists():
        doc_emb = np.load(doc_npy)
        log.info("")
        log.info("  doc_embeddings   shape: %s  dtype: %s", doc_emb.shape, doc_emb.dtype)

    if chunk_npy.exists():
        chunk_emb = np.load(chunk_npy)
        log.info("  chunk_embeddings shape: %s  dtype: %s", chunk_emb.shape, chunk_emb.dtype)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    processed_dir: Path,
    output_dir:    Path,
    batch_size:    int,
    chunk_size:    int,
    chunk_overlap: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()

    run_doc_pipeline(
        model         = model,
        processed_dir = processed_dir,
        output_dir    = output_dir,
        batch_size    = batch_size,
    )

    run_chunk_pipeline(
        model         = model,
        processed_dir = processed_dir,
        output_dir    = output_dir,
        batch_size    = batch_size,
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
    )

    print_summary(output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Patent embedding pipeline")
    parser.add_argument(
        "--processed-dir",
        default="../data/processed",
        help="Output directory from preprocessor.py (default: <script_dir>/data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/embeddings",
        help="Where to save embeddings and FAISS indexes (default: <script_dir>/data/embeddings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"SBERT encoding batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Words per RAG chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Word overlap between chunks (default: {CHUNK_OVERLAP})",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    output_dir    = Path(args.output_dir).resolve()

    log.info("Processed dir : %s", processed_dir)
    log.info("Output dir    : %s", output_dir)

    # Validate input directories exist before doing any work
    for subdir in ("doc_txt", "rag_txt", "cleaned_json"):
        p = processed_dir / subdir
        if not p.exists():
            raise FileNotFoundError(
                f"Expected directory not found: {p}\n"
                f"Make sure preprocessor.py has been run and completed successfully."
            )

    log.info("Input directories verified. Starting embedding pipeline...")

    run_pipeline(
        processed_dir = processed_dir,
        output_dir    = output_dir,
        batch_size    = args.batch_size,
        chunk_size    = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
    )