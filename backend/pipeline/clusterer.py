"""
Patent Clustering Pipeline
=========================
Implements two clustering tracks over the processed patent corpus:

1) Baseline (recommended first):
   TF-IDF (doc_txt) + K-Means

2) State-of-the-methods (optional advanced):
   SBERT doc embeddings + UMAP + HDBSCAN

Both tracks save per-document cluster assignments and a metrics summary.

Usage:
    python clusterer.py --method baseline
    python clusterer.py --method sota
    python clusterer.py --method both
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


class BM25Vectorizer:
    """
    Sparse BM25 document-term weighting.

    This uses CountVectorizer for tokenization and term counts, then applies
    Okapi BM25 weighting to each non-zero term frequency.
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        stop_words: str | None = "english",
        max_features: int = 12000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._counter = CountVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        self.k1 = k1
        self.b = b
        self.vocabulary_: dict[str, int] = {}

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        counts = self._counter.fit_transform(texts)
        self.vocabulary_ = self._counter.vocabulary_

        n_docs = counts.shape[0]
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        avg_doc_len = float(doc_lengths.mean()) if doc_lengths.size else 1.0
        if avg_doc_len <= 0:
            avg_doc_len = 1.0

        binary = counts.copy()
        binary.data = np.ones_like(binary.data)
        df = np.asarray(binary.sum(axis=0)).ravel().astype(np.float64)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        coo = counts.tocoo()
        tf = coo.data.astype(np.float64)
        denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_lengths[coo.row] / avg_doc_len))
        weights = idf[coo.col] * ((tf * (self.k1 + 1.0)) / denom)

        return csr_matrix((weights, (coo.row, coo.col)), shape=counts.shape)


def _load_text_corpus(processed_dir: Path) -> tuple[list[str], list[dict], list[str]]:
    """
    Loads doc_txt corpus for clustering and aligns metadata from cleaned_json.

    Returns:
      texts: list of patent text strings
      metadata: list of dictionaries with patent_id/title/domain
      y_true: list of true domain labels (for evaluation)
    """
    doc_txt_dir = processed_dir / "doc_txt"
    json_dir = processed_dir / "cleaned_json"

    if not doc_txt_dir.exists():
        raise FileNotFoundError(f"Missing directory: {doc_txt_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing directory: {json_dir}")

    texts: list[str] = []
    metadata: list[dict] = []
    y_true: list[str] = []

    for path in sorted(doc_txt_dir.glob("*.txt")):
        patent_id = path.stem
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        meta_path = json_dir / f"{patent_id}.json"
        if not meta_path.exists():
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        domain = m.get("domain", "unknown")
        title = m.get("title", "")

        texts.append(text)
        y_true.append(domain)
        metadata.append(
            {
                "patent_id": patent_id,
                "title": title,
                "domain": domain,
            }
        )

    if not texts:
        raise RuntimeError("No valid documents found in processed/doc_txt")

    return texts, metadata, y_true


def _safe_silhouette(features, labels: np.ndarray) -> float | None:
    n_clusters = len(set(labels))
    if n_clusters <= 1:
        return None
    return float(silhouette_score(features, labels))


def _cluster_purity(y_true: list[str], y_pred: np.ndarray) -> float:
    buckets: dict[int, list[str]] = defaultdict(list)
    for t, p in zip(y_true, y_pred):
        buckets[int(p)].append(t)

    majority = 0
    for members in buckets.values():
        majority += Counter(members).most_common(1)[0][1]

    return majority / max(len(y_true), 1)


def _save_outputs(
    output_dir: Path,
    method: str,
    metadata: list[dict],
    y_pred: np.ndarray,
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = []
    for i, row in enumerate(metadata):
        assignments.append(
            {
                "index": i,
                "patent_id": row["patent_id"],
                "domain_true": row["domain"],
                "cluster_pred": int(y_pred[i]),
                "title": row["title"],
            }
        )

    assignments_path = output_dir / f"{method}_cluster_assignments.json"
    with open(assignments_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)

    metrics_path = output_dir / f"{method}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log.info("Saved %s", assignments_path)
    log.info("Saved %s", metrics_path)


def run_baseline(
    processed_dir: Path,
    output_dir: Path,
    n_clusters: int,
    max_features: int,
    random_state: int,
    vectorizer_type: str,
    pca_components: int,
) -> None:
    log.info("Running baseline: %s + K-Means", vectorizer_type.upper())
    texts, metadata, y_true = _load_text_corpus(processed_dir)

    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
    elif vectorizer_type == "bm25":
        vectorizer = BM25Vectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
    else:
        raise ValueError(f"Unsupported vectorizer: {vectorizer_type}")

    x_sparse = vectorizer.fit_transform(texts)
    x_for_clustering = x_sparse
    projection: dict[str, object] = {
        "enabled": False,
        "method": "none",
    }

    if pca_components > 0:
        n_docs, n_features = x_sparse.shape
        max_components = min(pca_components, n_docs - 1, n_features)
        if max_components < 2:
            log.warning("Skipping PCA: not enough samples/features for requested components")
        else:
            dense_size = n_docs * n_features
            if dense_size <= 20_000_000:
                x_dense = x_sparse.toarray() if issparse(x_sparse) else np.asarray(x_sparse)
                pca = PCA(n_components=max_components, random_state=random_state)
                x_for_clustering = pca.fit_transform(x_dense)
                projection = {
                    "enabled": True,
                    "method": "pca",
                    "components": int(max_components),
                    "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                }
            else:
                # TruncatedSVD is still a linear projection and keeps sparse efficiency.
                svd = TruncatedSVD(n_components=max_components, random_state=random_state)
                x_for_clustering = svd.fit_transform(x_sparse)
                projection = {
                    "enabled": True,
                    "method": "truncated_svd_linear_projection",
                    "components": int(max_components),
                    "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
                }

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    y_pred = kmeans.fit_predict(x_for_clustering)

    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    hmg = float(homogeneity_score(y_true, y_pred))
    sil = _safe_silhouette(x_for_clustering, y_pred)
    purity = float(_cluster_purity(y_true, y_pred))

    metrics = {
        "method": f"baseline_{vectorizer_type}_kmeans",
        "num_documents": len(texts),
        "num_clusters": n_clusters,
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "projection": projection,
        "ari": ari,
        "nmi": nmi,
        "homogeneity": hmg,
        "silhouette": sil,
        "purity": purity,
    }

    _save_outputs(output_dir, "baseline", metadata, y_pred, metrics)
    log.info(
        "Baseline complete (%s). ARI=%.4f NMI=%.4f Purity=%.4f",
        vectorizer_type,
        ari,
        nmi,
        purity,
    )


def run_sota(
    embedding_dir: Path,
    output_dir: Path,
    random_state: int,
) -> None:
    """
    Advanced clustering path:
      doc_embeddings.npy -> UMAP -> HDBSCAN

    This path is optional and imports heavy dependencies lazily so baseline
    can run without them.
    """
    try:
        import umap
        import hdbscan
    except ImportError as e:
        raise RuntimeError(
            "SOTA mode requires umap-learn and hdbscan. Install both packages first."
        ) from e

    emb_path = embedding_dir / "doc_embeddings.npy"
    meta_path = embedding_dir / "doc_metadata.json"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path} or {meta_path}. Run embedder.py first."
        )

    x = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    reducer = umap.UMAP(
        n_components=15,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    x_reduced = reducer.fit_transform(x)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=4,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    y_pred = clusterer.fit_predict(x_reduced)

    n_noise = int(np.sum(y_pred == -1))
    n_clusters = int(len(set(y_pred)) - (1 if -1 in y_pred else 0))

    metrics = {
        "method": "sota_sbert_umap_hdbscan",
        "num_documents": int(x.shape[0]),
        "embedding_dim": int(x.shape[1]),
        "num_clusters": n_clusters,
        "noise_points": n_noise,
    }

    _save_outputs(output_dir, "sota", metadata, y_pred, metrics)
    log.info(
        "SOTA complete. clusters=%d noise=%d (%.2f%%)",
        n_clusters,
        n_noise,
        100.0 * n_noise / max(len(y_pred), 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Patent clustering pipeline")
    parser.add_argument(
        "--processed-dir",
        default=str(REPO_ROOT / "data" / "processed"),
        help="Directory from preprocessor output",
    )
    parser.add_argument(
        "--embedding-dir",
        default=str(REPO_ROOT / "data" / "embeddings"),
        help="Directory from embedder output",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "clustering"),
        help="Where to write cluster assignments and metrics",
    )
    parser.add_argument(
        "--method",
        choices=["baseline", "sota", "both"],
        default="baseline",
        help="Which algorithm track to run",
    )
    parser.add_argument("--n-clusters", type=int, default=5, help="K in baseline K-Means")
    parser.add_argument(
        "--max-features",
        type=int,
        default=12000,
        help="Max TF-IDF features in baseline mode",
    )
    parser.add_argument(
        "--vectorizer",
        choices=["tfidf", "bm25"],
        default="tfidf",
        help="Keyword vectorizer for baseline mode",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="Linear projection dimensions before K-Means (0 disables PCA)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    embedding_dir = Path(args.embedding_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    log.info("processed_dir=%s", processed_dir)
    log.info("embedding_dir=%s", embedding_dir)
    log.info("output_dir=%s", output_dir)

    if args.method in {"baseline", "both"}:
        run_baseline(
            processed_dir=processed_dir,
            output_dir=output_dir,
            n_clusters=args.n_clusters,
            max_features=args.max_features,
            random_state=args.seed,
            vectorizer_type=args.vectorizer,
            pca_components=args.pca_components,
        )

    if args.method in {"sota", "both"}:
        run_sota(
            embedding_dir=embedding_dir,
            output_dir=output_dir,
            random_state=args.seed,
        )


if __name__ == "__main__":
    main()
