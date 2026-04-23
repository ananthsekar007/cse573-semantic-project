from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "backend" / "data"
CLUSTERING_DIR = DATA_ROOT / "clustering"


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_cluster_dashboard() -> dict:
    assignments = _load_json(CLUSTERING_DIR / "umap_hdbscan_assignments.json")
    metrics = _load_json(CLUSTERING_DIR / "umap_hdbscan_metrics.json")
    labels = _load_json(CLUSTERING_DIR / "umap_hdbscan_cluster_labels.json")

    cluster_points: dict[int, list[dict]] = defaultdict(list)
    for point in assignments:
        cluster_points[int(point["cluster_pred"])].append(point)

    clusters = []
    for cluster_id in sorted(
        (int(key) for key in labels.keys()),
        key=lambda value: value,
    ):
        label_info = labels[str(cluster_id)]
        points = cluster_points.get(cluster_id, [])
        domain_counter = Counter(point.get("domain_true", "unknown") for point in points)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": label_info.get("label", f"Cluster {cluster_id}"),
                "keywords": label_info.get("keywords", []),
                "tfidf_terms": label_info.get("tfidf_terms", []),
                "num_patents": label_info.get("num_patents", len(points)),
                "dominant_domain": label_info.get("dominant_domain", "unknown"),
                "domain_breakdown": label_info.get("domain_breakdown", dict(domain_counter)),
                "sample_patents": label_info.get("sample_patents", []),
                "color": points[0]["color"] if points else "#999999",
            }
        )

    summary = {
        "num_documents": metrics.get("num_documents", len(assignments)),
        "num_clusters": metrics.get("num_clusters", len(clusters)),
        "noise_points": metrics.get("noise_points", 0),
        "ari": metrics.get("ari"),
        "nmi": metrics.get("nmi"),
        "homogeneity": metrics.get("homogeneity"),
        "purity": metrics.get("purity"),
        "silhouette": metrics.get("silhouette"),
    }

    return {
        "summary": summary,
        "metrics": metrics,
        "clusters": clusters,
        "points": assignments,
    }
