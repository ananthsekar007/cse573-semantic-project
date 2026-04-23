"""
UMAP + HDBSCAN cluster visualizer for patent embeddings.

Reads document embeddings from backend/data/embeddings, clusters them with
HDBSCAN in embedding space, projects them to 2D with UMAP for display, and writes:

  - umap_hdbscan_assignments.json  per-patent cluster membership
  - umap_hdbscan_metrics.json      summary metrics
  - umap_hdbscan_projection.npy    2D UMAP coordinates
  - umap_hdbscan_visualization.html self-contained interactive scatter plot

Usage:
    python backend/pipeline/umap_hdbscan_visualizer.py
    python backend/pipeline/umap_hdbscan_visualizer.py --embedding-dir backend/data/embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import umap
import hdbscan
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "backend" / "data" / "embeddings"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "backend" / "data" / "clustering"

CLUSTER_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
NOISE_COLOR = "#b8b8b8"


def load_embeddings(embedding_dir: Path) -> tuple[np.ndarray, list[dict]]:
    emb_path = embedding_dir / "doc_embeddings.npy"
    meta_path = embedding_dir / "doc_metadata.json"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    embeddings = np.load(emb_path).astype(np.float32)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape {embeddings.shape}")
    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Metadata count ({len(metadata)}) does not match embeddings count ({len(embeddings)})"
        )

    return embeddings, metadata


def cluster_purity(y_true: list[str], y_pred: np.ndarray) -> float:
    buckets: dict[int, list[str]] = defaultdict(list)
    for true_label, pred_label in zip(y_true, y_pred):
        if int(pred_label) == -1:
            continue
        buckets[int(pred_label)].append(true_label)

    assigned = sum(len(members) for members in buckets.values())
    if assigned == 0:
        return 0.0

    majority_total = 0
    for members in buckets.values():
        majority_total += Counter(members).most_common(1)[0][1]

    return majority_total / assigned


def safe_silhouette(points: np.ndarray, labels: np.ndarray) -> float | None:
    non_noise_mask = labels != -1
    if int(non_noise_mask.sum()) < 3:
        return None

    filtered_points = points[non_noise_mask]
    filtered_labels = labels[non_noise_mask]
    if len(set(filtered_labels.tolist())) <= 1:
        return None

    return float(silhouette_score(filtered_points, filtered_labels))


def truncate_title(title: str, limit: int = 120) -> str:
    title = " ".join(title.split())
    if len(title) <= limit:
        return title
    return title[: limit - 1].rstrip() + "…"


def cluster_color(cluster_id: int) -> str:
    if cluster_id == -1:
        return NOISE_COLOR
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def assign_noise_points(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Assign HDBSCAN noise points to the nearest discovered cluster centroid."""
    filled = labels.copy()
    noise_indices = np.where(labels == -1)[0]
    non_noise_clusters = [cluster_id for cluster_id in sorted(set(labels.tolist())) if cluster_id != -1]
    if len(noise_indices) == 0 or not non_noise_clusters:
        return filled, []

    centroids: dict[int, np.ndarray] = {}
    for cluster_id in non_noise_clusters:
        members = embeddings[labels == cluster_id]
        centroid = members.mean(axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        centroids[cluster_id] = centroid

    reassigned: list[int] = []
    for idx in noise_indices:
        vec = embeddings[idx : idx + 1]
        similarities = [
            (cluster_id, float(cosine_similarity(vec, centroid)[0, 0]))
            for cluster_id, centroid in centroids.items()
        ]
        similarities.sort(key=lambda item: item[1], reverse=True)
        filled[idx] = similarities[0][0]
        reassigned.append(int(idx))

    return filled, reassigned


def build_html(points: list[dict], metrics: dict) -> str:
    points_json = json.dumps(points, ensure_ascii=False)
    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Patent UMAP + HDBSCAN Clusters</title>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: #fffdf8;
      --ink: #1e1b18;
      --muted: #6f685f;
      --grid: #ddd3c6;
      --accent: #bf5b04;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(191, 91, 4, 0.10), transparent 26%),
        radial-gradient(circle at bottom right, rgba(31, 119, 180, 0.10), transparent 24%),
        linear-gradient(180deg, #f8f4ed, var(--bg));
    }}
    .wrap {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .header {{
      display: grid;
      gap: 8px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.8rem, 4vw, 3rem);
      line-height: 1.04;
      letter-spacing: -0.03em;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      max-width: 70ch;
      font-size: 1rem;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 18px 0 22px;
    }}
    .metric {{
      background: rgba(255, 253, 248, 0.88);
      border: 1px solid rgba(111, 104, 95, 0.18);
      border-radius: 16px;
      padding: 14px 16px;
      backdrop-filter: blur(4px);
      box-shadow: 0 8px 26px rgba(50, 38, 24, 0.06);
    }}
    .metric .label {{
      display: block;
      font-size: 0.82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric .value {{
      display: block;
      margin-top: 6px;
      font-size: 1.45rem;
      font-weight: 700;
    }}
    .panel {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid rgba(111, 104, 95, 0.18);
      border-radius: 20px;
      overflow: hidden;
      box-shadow: 0 14px 40px rgba(50, 38, 24, 0.08);
    }}
    .panel-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: center;
      justify-content: space-between;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(111, 104, 95, 0.14);
    }}
    .panel-head p {{
      margin: 0;
      color: var(--muted);
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(244, 240, 232, 0.95);
      border-radius: 999px;
      padding: 5px 10px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
      border: 1px solid rgba(0, 0, 0, 0.18);
    }}
    .chart-wrap {{
      position: relative;
      padding: 16px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.8), rgba(250,247,241,0.96));
      border-radius: 16px;
    }}
    .tooltip {{
      position: absolute;
      min-width: 240px;
      max-width: 320px;
      pointer-events: none;
      opacity: 0;
      transform: translate(-50%, calc(-100% - 14px));
      background: rgba(28, 24, 20, 0.95);
      color: #fff9f0;
      border-radius: 12px;
      padding: 10px 12px;
      box-shadow: 0 16px 38px rgba(0, 0, 0, 0.22);
      transition: opacity 120ms ease;
      font-family: ui-sans-serif, system-ui, sans-serif;
      font-size: 0.88rem;
      line-height: 1.4;
    }}
    .tooltip strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 0.92rem;
    }}
    .foot {{
      padding: 14px 18px 18px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.92em;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>Patent Cluster Map</h1>
      <p class="subtitle">
        UMAP projection of the document embeddings, with HDBSCAN cluster assignments.
        Hover any point to inspect the patent metadata used in the demo corpus.
      </p>
    </div>

    <div class="metrics" id="metrics"></div>

    <div class="panel">
      <div class="panel-head">
        <p>Each point is one patent embedding from <code>doc_embeddings.npy</code>.</p>
        <div class="legend" id="legend"></div>
      </div>
      <div class="chart-wrap">
        <svg id="chart" viewBox="0 0 1100 760" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="tooltip" id="tooltip"></div>
      </div>
      <div class="foot">
        Noise points are marked as cluster <code>-1</code>. Metrics shown here are computed from
        the same clustering run and the known domain labels already present in the metadata.
      </div>
    </div>
  </div>

  <script>
    const points = {points_json};
    const metrics = {metrics_json};

    const svg = document.getElementById("chart");
    const tooltip = document.getElementById("tooltip");
    const metricsEl = document.getElementById("metrics");
    const legendEl = document.getElementById("legend");

    const width = 1100;
    const height = 760;
    const margin = 70;

    const minX = Math.min(...points.map(p => p.umap_x));
    const maxX = Math.max(...points.map(p => p.umap_x));
    const minY = Math.min(...points.map(p => p.umap_y));
    const maxY = Math.max(...points.map(p => p.umap_y));

    const scaleX = value => {{
      if (maxX === minX) return width / 2;
      return margin + ((value - minX) / (maxX - minX)) * (width - 2 * margin);
    }};
    const scaleY = value => {{
      if (maxY === minY) return height / 2;
      const t = (value - minY) / (maxY - minY);
      return height - margin - t * (height - 2 * margin);
    }};

    function fmt(value) {{
      if (value === null || value === undefined) return "N/A";
      if (typeof value === "number") return Number(value).toFixed(4);
      return String(value);
    }}

    const metricCards = [
      ["Documents", metrics.num_documents],
      ["Clusters", metrics.num_clusters],
      ["Noise Points", metrics.noise_points],
      ["ARI", metrics.ari],
      ["NMI", metrics.nmi],
      ["Homogeneity", metrics.homogeneity],
      ["Purity", metrics.purity],
      ["Silhouette", metrics.silhouette],
    ];

    metricsEl.innerHTML = metricCards.map(([label, value]) => `
      <div class="metric">
        <span class="label">${{label}}</span>
        <span class="value">${{fmt(value)}}</span>
      </div>
    `).join("");

    const clusterIds = [...new Set(points.map(p => p.cluster_pred))].sort((a, b) => a - b);
    legendEl.innerHTML = clusterIds.map(cid => `
      <span class="legend-item">
        <span class="swatch" style="background:${{points.find(p => p.cluster_pred === cid).color}}"></span>
        Cluster ${{cid}}
      </span>
    `).join("");

    const svgns = "http://www.w3.org/2000/svg";

    function addLine(x1, y1, x2, y2, stroke, widthValue, dash = "") {{
      const line = document.createElementNS(svgns, "line");
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      line.setAttribute("stroke", stroke);
      line.setAttribute("stroke-width", widthValue);
      if (dash) line.setAttribute("stroke-dasharray", dash);
      svg.appendChild(line);
    }}

    function addText(x, y, text, fill, anchor = "middle", size = "13") {{
      const node = document.createElementNS(svgns, "text");
      node.setAttribute("x", x);
      node.setAttribute("y", y);
      node.setAttribute("fill", fill);
      node.setAttribute("font-size", size);
      node.setAttribute("font-family", "ui-sans-serif, system-ui, sans-serif");
      node.setAttribute("text-anchor", anchor);
      node.textContent = text;
      svg.appendChild(node);
    }}

    for (let i = 0; i < 5; i += 1) {{
      const x = margin + ((width - 2 * margin) / 4) * i;
      addLine(x, margin, x, height - margin, "rgba(110,104,95,0.22)", 1, "4 8");
    }}
    for (let i = 0; i < 5; i += 1) {{
      const y = margin + ((height - 2 * margin) / 4) * i;
      addLine(margin, y, width - margin, y, "rgba(110,104,95,0.22)", 1, "4 8");
    }}

    addLine(margin, height - margin, width - margin, height - margin, "#7b7268", 1.5);
    addLine(margin, margin, margin, height - margin, "#7b7268", 1.5);
    addText(width / 2, height - 20, "UMAP-1", "#6f685f");
    addText(24, height / 2, "UMAP-2", "#6f685f");

    points.forEach(point => {{
      const circle = document.createElementNS(svgns, "circle");
      circle.setAttribute("cx", scaleX(point.umap_x));
      circle.setAttribute("cy", scaleY(point.umap_y));
      circle.setAttribute("r", point.cluster_pred === -1 ? 5.5 : 7);
      circle.setAttribute("fill", point.color);
      circle.setAttribute("fill-opacity", point.cluster_pred === -1 ? "0.55" : "0.82");
      circle.setAttribute("stroke", point.cluster_pred === -1 ? "#6f685f" : "rgba(20,20,20,0.45)");
      circle.setAttribute("stroke-width", point.cluster_pred === -1 ? "1.2" : "0.8");
      circle.style.cursor = "pointer";

      circle.addEventListener("mouseenter", event => {{
        tooltip.style.opacity = "1";
        tooltip.innerHTML = `
          <strong>${{point.patent_id}}</strong>
          <div>${{point.title}}</div>
          <div>True domain: ${{point.domain_true}}</div>
          <div>Cluster: ${{point.cluster_pred}}</div>
          <div>UMAP: (${{point.umap_x.toFixed(3)}}, ${{point.umap_y.toFixed(3)}})</div>
        `;
      }});

      circle.addEventListener("mousemove", event => {{
        const bounds = svg.getBoundingClientRect();
        const wrapBounds = svg.parentElement.getBoundingClientRect();
        tooltip.style.left = `${{event.clientX - wrapBounds.left}}px`;
        tooltip.style.top = `${{event.clientY - wrapBounds.top}}px`;
      }});

      circle.addEventListener("mouseleave", () => {{
        tooltip.style.opacity = "0";
      }});

      svg.appendChild(circle);
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP + HDBSCAN visualizer for patent embeddings")
    parser.add_argument(
        "--embedding-dir",
        default=str(DEFAULT_EMBEDDING_DIR),
        help="Directory containing doc_embeddings.npy and doc_metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to save clustering artifacts and the HTML visualizer",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UMAP")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.0, help="UMAP min_dist")
    parser.add_argument("--min-cluster-size", type=int, default=4, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=1, help="HDBSCAN min_samples")
    parser.add_argument(
        "--cluster-selection-method",
        choices=["eom", "leaf"],
        default="leaf",
        help="HDBSCAN cluster selection method",
    )
    parser.add_argument(
        "--assign-noise",
        action="store_true",
        default=True,
        help="Assign HDBSCAN noise points to the nearest discovered cluster centroid",
    )
    parser.add_argument(
        "--no-assign-noise",
        dest="assign_noise",
        action="store_false",
        help="Keep HDBSCAN noise points as cluster -1",
    )
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_embeddings(embedding_dir)
    y_true = [row.get("domain", "unknown") for row in metadata]

    log.info("Loaded %d document embeddings from %s", len(embeddings), embedding_dir)
    log.info("Running HDBSCAN clustering in embedding space...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method=args.cluster_selection_method,
        prediction_data=True,
    )
    y_pred_raw = clusterer.fit_predict(embeddings)
    reassigned_indices: list[int] = []
    y_pred = y_pred_raw.copy()
    if args.assign_noise:
        y_pred, reassigned_indices = assign_noise_points(embeddings, y_pred_raw)

    log.info("Running UMAP projection for visualization...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    points_2d = reducer.fit_transform(embeddings)

    n_noise_raw = int(np.sum(y_pred_raw == -1))
    n_noise_final = int(np.sum(y_pred == -1))
    n_clusters = int(len(set(y_pred.tolist())) - (1 if -1 in y_pred else 0))

    metrics = {
        "method": "embedding_hdbscan_plus_umap_visualizer",
        "num_documents": int(len(metadata)),
        "num_clusters": n_clusters,
        "noise_points": n_noise_final,
        "raw_noise_points": n_noise_raw,
        "noise_reassigned": len(reassigned_indices),
        "umap_neighbors": int(args.umap_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples),
        "cluster_selection_method": args.cluster_selection_method,
        "clustering_space": "original_embeddings",
        "visualization_space": "umap_2d",
        "assign_noise": bool(args.assign_noise),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "purity": float(cluster_purity(y_true, y_pred)),
        "silhouette": safe_silhouette(points_2d, y_pred),
    }

    assignments = []
    plot_points = []
    for idx, (row, coords, cluster_id) in enumerate(zip(metadata, points_2d, y_pred)):
        point = {
            "index": idx,
            "patent_id": row.get("patent_id", f"row_{idx}"),
            "title": truncate_title(row.get("title", "")),
            "domain_true": row.get("domain", "unknown"),
            "cluster_pred": int(cluster_id),
            "raw_cluster_pred": int(y_pred_raw[idx]),
            "was_noise": bool(y_pred_raw[idx] == -1 and cluster_id != -1),
            "umap_x": float(coords[0]),
            "umap_y": float(coords[1]),
            "color": cluster_color(int(cluster_id)),
        }
        assignments.append(point)
        plot_points.append(point)

    assignments_path = output_dir / "umap_hdbscan_assignments.json"
    metrics_path = output_dir / "umap_hdbscan_metrics.json"
    projection_path = output_dir / "umap_hdbscan_projection.npy"
    html_path = output_dir / "umap_hdbscan_visualization.html"

    with open(assignments_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.save(projection_path, points_2d.astype(np.float32))
    html_path.write_text(build_html(plot_points, metrics), encoding="utf-8")

    log.info("Saved %s", assignments_path)
    log.info("Saved %s", metrics_path)
    log.info("Saved %s", projection_path)
    log.info("Saved %s", html_path)
    log.info(
        "Complete. clusters=%d noise=%d ari=%.4f nmi=%.4f purity=%.4f",
        n_clusters,
        n_noise_final,
        metrics["ari"],
        metrics["nmi"],
        metrics["purity"],
    )


if __name__ == "__main__":
    main()
