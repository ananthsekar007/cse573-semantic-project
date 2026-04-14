"""
Novelty Scoring Baseline
========================
Computes novelty scores for patents by comparing document embeddings against the
existing FAISS doc index.

Baseline logic:
- Use the document-level SBERT embeddings produced by embedder.py
- Build or load a FAISS cosine-similarity index
- For each patent, find the nearest prior-art neighbor
- Novelty score = 1 - max_similarity_to_other_documents

The script also supports ad hoc novelty queries for a new patent text.

Usage:
    python novelty.py
    python novelty.py --query-file ./candidate_patent.txt
    python novelty.py --query-text "A device for ..."
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768


@dataclass
class NeighborMatch:
    rank: int
    patent_id: str
    title: str
    domain: str
    similarity: float
    novelty_score: float


def load_model() -> SentenceTransformer:
    log.info("Loading SBERT model: %s", SBERT_MODEL)
    return SentenceTransformer(SBERT_MODEL, device="cpu")


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    vector = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return vector


def _load_embeddings(embedding_dir: Path) -> tuple[np.ndarray, list[dict]]:
    emb_path = embedding_dir / "doc_embeddings.npy"
    meta_path = embedding_dir / "doc_metadata.json"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Metadata count ({len(metadata)}) does not match embeddings count ({len(embeddings)})"
        )

    return embeddings.astype(np.float32), metadata


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix")
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    return index


def _load_or_build_index(embedding_dir: Path, embeddings: np.ndarray, refresh: bool) -> faiss.IndexFlatIP:
    index_path = embedding_dir / "faiss_doc.index"
    if index_path.exists() and not refresh:
        try:
            index = faiss.read_index(str(index_path))
            if index.ntotal == len(embeddings):
                return index
            log.warning(
                "FAISS index size (%d) does not match embeddings (%d); rebuilding",
                index.ntotal,
                len(embeddings),
            )
        except Exception as exc:
            log.warning("Failed to load FAISS index from %s: %s; rebuilding", index_path, exc)

    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(index_path))
    log.info("Saved FAISS doc index to %s", index_path)
    return index


def _pairwise_neighbors(
    embeddings: np.ndarray,
    metadata: list[dict],
    top_k: int,
) -> list[NeighborMatch]:
    index = build_faiss_index(embeddings)
    search_k = min(max(top_k + 1, 2), len(metadata))
    distances, neighbors = index.search(embeddings, search_k)

    matches: list[NeighborMatch] = []
    for row_idx, row_meta in enumerate(metadata):
        patent_id = row_meta.get("patent_id", f"row_{row_idx}")
        title = row_meta.get("title", "")
        domain = row_meta.get("domain", "unknown")

        chosen_rank = None
        chosen_neighbor = None
        chosen_similarity = None
        for rank, (neighbor_idx, similarity) in enumerate(zip(neighbors[row_idx], distances[row_idx]), start=1):
            if neighbor_idx == row_idx:
                continue
            chosen_rank = rank
            chosen_neighbor = metadata[int(neighbor_idx)]
            chosen_similarity = float(similarity)
            break

        if chosen_neighbor is None:
            chosen_rank = 1
            chosen_neighbor = row_meta
            chosen_similarity = float(distances[row_idx][0])

        matches.append(
            NeighborMatch(
                rank=int(chosen_rank),
                patent_id=patent_id,
                title=title,
                domain=domain,
                similarity=chosen_similarity,
                novelty_score=float(1.0 - chosen_similarity),
            )
        )

    return matches


def score_corpus_novelty(
    embedding_dir: Path,
    output_dir: Path,
    top_k: int,
    refresh_index: bool,
) -> dict:
    embeddings, metadata = _load_embeddings(embedding_dir)
    index = _load_or_build_index(embedding_dir, embeddings, refresh_index)

    search_k = min(max(top_k + 1, 2), len(metadata))
    distances, neighbors = index.search(embeddings, search_k)

    records: list[dict] = []
    similarities: list[float] = []
    novelty_scores: list[float] = []

    for row_idx, row_meta in enumerate(metadata):
        patent_id = row_meta.get("patent_id", f"row_{row_idx}")
        title = row_meta.get("title", "")
        domain = row_meta.get("domain", "unknown")

        best = None
        for neighbor_idx, similarity in zip(neighbors[row_idx], distances[row_idx]):
            if neighbor_idx == row_idx:
                continue
            best = (int(neighbor_idx), float(similarity))
            break

        if best is None:
            best = (row_idx, float(distances[row_idx][0]))

        best_idx, best_similarity = best
        best_meta = metadata[best_idx]
        novelty_score = float(1.0 - best_similarity)

        similarities.append(best_similarity)
        novelty_scores.append(novelty_score)

        records.append(
            {
                "index": row_idx,
                "patent_id": patent_id,
                "title": title,
                "domain": domain,
                "nearest_neighbor": {
                    "patent_id": best_meta.get("patent_id", f"row_{best_idx}"),
                    "title": best_meta.get("title", ""),
                    "domain": best_meta.get("domain", "unknown"),
                    "similarity": best_similarity,
                },
                "novelty_score": novelty_score,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "novelty_scores.json"
    summary_path = output_dir / "novelty_summary.json"

    summary = {
        "method": "faiss_doc_cosine_novelty",
        "num_documents": len(metadata),
        "top_k": top_k,
        "average_top1_similarity": float(np.mean(similarities)) if similarities else None,
        "average_novelty_score": float(np.mean(novelty_scores)) if novelty_scores else None,
        "median_novelty_score": float(median(novelty_scores)) if novelty_scores else None,
        "min_novelty_score": float(np.min(novelty_scores)) if novelty_scores else None,
        "max_novelty_score": float(np.max(novelty_scores)) if novelty_scores else None,
    }

    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("Saved %s", scores_path)
    log.info("Saved %s", summary_path)
    log.info(
        "Novelty baseline complete. avg_similarity=%.4f avg_novelty=%.4f",
        summary["average_top1_similarity"] or 0.0,
        summary["average_novelty_score"] or 0.0,
    )

    return {"summary": summary, "records": records}


def score_query(
    query_text: str,
    embedding_dir: Path,
    top_k: int,
    refresh_index: bool,
) -> dict:
    embeddings, metadata = _load_embeddings(embedding_dir)
    index = _load_or_build_index(embedding_dir, embeddings, refresh_index)
    model = load_model()
    query_vec = embed_text(model, query_text)
    search_k = min(max(top_k, 1), len(metadata))
    distances, neighbors = index.search(query_vec, search_k)

    matches = []
    for rank, (neighbor_idx, similarity) in enumerate(zip(neighbors[0], distances[0]), start=1):
        neighbor_meta = metadata[int(neighbor_idx)]
        matches.append(
            {
                "rank": rank,
                "patent_id": neighbor_meta.get("patent_id", f"row_{neighbor_idx}"),
                "title": neighbor_meta.get("title", ""),
                "domain": neighbor_meta.get("domain", "unknown"),
                "similarity": float(similarity),
                "novelty_score": float(1.0 - similarity),
            }
        )

    result = {
        "query_text": query_text,
        "top_k": top_k,
        "matches": matches,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Patent novelty scoring baseline")
    parser.add_argument(
        "--embedding-dir",
        default=str(REPO_ROOT / "data" / "embeddings"),
        help="Directory containing doc_embeddings.npy, doc_metadata.json, and faiss_doc.index",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "novelty"),
        help="Where to save novelty outputs",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many neighbors to report")
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="Rebuild the FAISS doc index from embeddings",
    )
    parser.add_argument("--query-text", type=str, default="", help="Score a new candidate patent text")
    parser.add_argument("--query-file", type=str, default="", help="Path to a file containing candidate text")
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.query_text or args.query_file:
        if args.query_file:
            query_text = Path(args.query_file).read_text(encoding="utf-8", errors="ignore")
        else:
            query_text = args.query_text
        result = score_query(query_text, embedding_dir, args.top_k, args.refresh_index)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    result = score_corpus_novelty(
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        top_k=args.top_k,
        refresh_index=args.refresh_index,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
