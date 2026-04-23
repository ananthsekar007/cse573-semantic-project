"""
Cluster labeling pipeline.

Generates human-readable cluster labels from an assignments file by combining:
  - KeyBERT keyphrases over the cluster's concatenated patent texts
  - TF-IDF distinctive terms for extra interpretability

Supported assignment artifacts include:
  - baseline_cluster_assignments.json
  - sota_cluster_assignments.json
  - umap_hdbscan_assignments.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

METHOD_TO_ASSIGNMENTS = {
    "baseline": "baseline_cluster_assignments.json",
    "sota": "sota_cluster_assignments.json",
    "umap_hdbscan": "umap_hdbscan_assignments.json",
}


def resolve_assignments_path(clustering_dir: Path, method: str, assignments_file: str) -> Path:
    if assignments_file:
        return Path(assignments_file).resolve()
    filename = METHOD_TO_ASSIGNMENTS.get(method, f"{method}_cluster_assignments.json")
    return clustering_dir / filename


def load_cluster_assignments(assignments_path: Path) -> list[dict]:
    with open(assignments_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_patent_text(doc_txt_dir: Path, patent_id: str) -> str:
    path = doc_txt_dir / f"{patent_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def group_texts_by_cluster(assignments: list[dict], doc_txt_dir: Path) -> dict[int, list[str]]:
    cluster_texts: dict[int, list[str]] = defaultdict(list)
    for item in assignments:
        cluster_id = int(item["cluster_pred"])
        if cluster_id == -1:
            continue
        text = load_patent_text(doc_txt_dir, item["patent_id"])
        if text:
            cluster_texts[cluster_id].append(text)
    return cluster_texts


def summarize_cluster_members(assignments: list[dict]) -> dict[int, dict]:
    summary: dict[int, dict] = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in assignments:
        cluster_id = int(item["cluster_pred"])
        if cluster_id == -1:
            continue
        grouped[cluster_id].append(item)

    for cluster_id, members in grouped.items():
        domain_counter = Counter(m.get("domain_true", "unknown") for m in members)
        summary[cluster_id] = {
            "cluster_id": cluster_id,
            "num_patents": len(members),
            "domain_breakdown": dict(domain_counter),
            "dominant_domain": domain_counter.most_common(1)[0][0] if domain_counter else "unknown",
            "sample_patents": [
                {
                    "patent_id": m.get("patent_id", ""),
                    "title": m.get("title", ""),
                    "domain_true": m.get("domain_true", "unknown"),
                }
                for m in members[:5]
            ],
        }
    return summary


def generate_labels(cluster_texts: dict[int, list[str]], top_n: int = 5, try_keybert: bool = False) -> dict[int, dict]:
    kw_model: KeyBERT | None = None
    use_keybert = try_keybert

    try:
        if not try_keybert:
            raise RuntimeError("KeyBERT disabled for offline/local execution")
        log.info("Loading KeyBERT model...")
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    except Exception as exc:
        use_keybert = False
        log.warning("KeyBERT model unavailable, falling back to TF-IDF-only labels: %s", exc)

    labels: dict[int, dict] = {}
    for cluster_id in sorted(cluster_texts.keys()):
        texts = cluster_texts[cluster_id]
        log.info("Labeling cluster %d (%d patents)...", cluster_id, len(texts))
        combined = " ".join(texts)[:10_000]

        if use_keybert and kw_model is not None:
            keywords = kw_model.extract_keywords(
                combined,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_n,
                diversity=0.5,
            )
            keyword_strings = [kw for kw, _score in keywords]
            keyword_scores = {kw: round(score, 4) for kw, score in keywords}
        else:
            keywords = []
            keyword_strings = []
            keyword_scores = {}

        label = " · ".join(keyword_strings[:3]).title() if keyword_strings else f"Cluster {cluster_id}"
        labels[cluster_id] = {
            "cluster_id": cluster_id,
            "num_patents": len(texts),
            "label": label,
            "keywords": keyword_strings,
            "keyword_scores": keyword_scores,
        }
        log.info("  -> %s", label)
    return labels


def add_tfidf_terms(labels: dict[int, dict], cluster_texts: dict[int, list[str]], top_n: int = 5) -> dict[int, dict]:
    log.info("Computing TF-IDF distinctive terms...")
    cluster_ids = sorted(cluster_texts.keys())
    docs = [" ".join(cluster_texts[cid])[:10_000] for cid in cluster_ids]
    if len(docs) < 2:
        return labels

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()

    for i, cluster_id in enumerate(cluster_ids):
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        tfidf_terms = [features[j] for j in top_idx if row[j] > 0]
        labels[cluster_id]["tfidf_terms"] = tfidf_terms
        if not labels[cluster_id]["keywords"]:
            labels[cluster_id]["keywords"] = tfidf_terms
            labels[cluster_id]["label"] = " · ".join(tfidf_terms[:3]).title() if tfidf_terms else labels[cluster_id]["label"]
    return labels


def merge_cluster_summary(labels: dict[int, dict], cluster_summary: dict[int, dict]) -> dict[int, dict]:
    for cluster_id, info in labels.items():
        info.update(cluster_summary.get(cluster_id, {}))
    return labels


def print_summary(labels: dict[int, dict]) -> None:
    print("\n" + "=" * 60)
    print("CLUSTER LABELS SUMMARY")
    print("=" * 60)
    for cluster_id in sorted(labels.keys()):
        info = labels[cluster_id]
        print(f"\nCluster {cluster_id} ({info['num_patents']} patents)")
        print(f"  Label    : {info['label']}")
        print(f"  Keywords : {', '.join(info['keywords'])}")
        if info.get("tfidf_terms"):
            print(f"  TF-IDF   : {', '.join(info['tfidf_terms'])}")
        if info.get("dominant_domain"):
            print(f"  Domain   : {info['dominant_domain']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cluster labels with KeyBERT + TF-IDF")
    parser.add_argument(
        "--method",
        choices=["baseline", "sota", "umap_hdbscan"],
        default="baseline",
        help="Cluster artifact family to label",
    )
    parser.add_argument(
        "--assignments-file",
        default="",
        help="Optional explicit path to a cluster assignments JSON file",
    )
    parser.add_argument("--clustering-dir", default=str(REPO_ROOT / "backend" / "data" / "clustering"))
    parser.add_argument("--processed-dir", default=str(REPO_ROOT / "backend" / "data" / "processed"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "backend" / "data" / "clustering"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--try-keybert",
        action="store_true",
        help="Attempt true KeyBERT labeling. Leave off for offline/local TF-IDF fallback labels.",
    )
    args = parser.parse_args()

    clustering_dir = Path(args.clustering_dir).resolve()
    doc_txt_dir = Path(args.processed_dir).resolve() / "doc_txt"
    output_dir = Path(args.output_dir).resolve()
    assignments_path = resolve_assignments_path(clustering_dir, args.method, args.assignments_file)

    assignments = load_cluster_assignments(assignments_path)
    log.info("Loaded %d assignments from %s", len(assignments), assignments_path)

    cluster_texts = group_texts_by_cluster(assignments, doc_txt_dir)
    log.info("Found %d non-noise clusters", len(cluster_texts))

    labels = generate_labels(cluster_texts, top_n=args.top_n, try_keybert=args.try_keybert)
    labels = add_tfidf_terms(labels, cluster_texts, top_n=args.top_n)
    labels = merge_cluster_summary(labels, summarize_cluster_members(assignments))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.method}_cluster_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2, ensure_ascii=False)
    log.info("Saved -> %s", out_path)

    print_summary(labels)


if __name__ == "__main__":
    main()
