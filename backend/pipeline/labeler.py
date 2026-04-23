"""
Cluster Labeling Pipeline
"""
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

def load_cluster_assignments(clustering_dir, method):
    path = clustering_dir / f"{method}_cluster_assignments.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_patent_text(doc_txt_dir, patent_id):
    path = doc_txt_dir / f"{patent_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()

def group_texts_by_cluster(assignments, doc_txt_dir):
    cluster_texts = defaultdict(list)
    for item in assignments:
        cluster_id = item["cluster_pred"]
        if cluster_id == -1:
            continue
        text = load_patent_text(doc_txt_dir, item["patent_id"])
        if text:
            cluster_texts[cluster_id].append(text)
    return cluster_texts

def generate_labels(cluster_texts, top_n=5):
    log.info("Loading KeyBERT model...")
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    labels = {}
    for cluster_id in sorted(cluster_texts.keys()):
        texts = cluster_texts[cluster_id]
        log.info("Labeling cluster %d (%d patents)...", cluster_id, len(texts))
        combined = " ".join(texts)[:10_000]
        keywords = kw_model.extract_keywords(combined, keyphrase_ngram_range=(1,2), stop_words="english", top_n=top_n, diversity=0.5)
        keyword_strings = [kw for kw, score in keywords]
        label = " · ".join(keyword_strings[:3]).title()
        labels[cluster_id] = {
            "cluster_id": cluster_id,
            "num_patents": len(texts),
            "label": label,
            "keywords": keyword_strings,
            "keyword_scores": {kw: round(score, 4) for kw, score in keywords},
        }
        log.info("  → %s", label)
    return labels

def add_tfidf_terms(labels, cluster_texts, top_n=5):
    log.info("Computing TF-IDF distinctive terms...")
    cluster_ids = sorted(cluster_texts.keys())
    docs = [" ".join(cluster_texts[cid])[:10_000] for cid in cluster_ids]
    if len(docs) < 2:
        return labels
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2), min_df=1)
    matrix = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()
    for i, cid in enumerate(cluster_ids):
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        labels[cid]["tfidf_terms"] = [features[j] for j in top_idx if row[j] > 0]
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["baseline","sota"], default="baseline")
    parser.add_argument("--clustering-dir", default=str(REPO_ROOT/"data"/"clustering"))
    parser.add_argument("--processed-dir", default=str(REPO_ROOT/"data"/"processed"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT/"data"/"clustering"))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    clustering_dir = Path(args.clustering_dir).resolve()
    doc_txt_dir = Path(args.processed_dir).resolve() / "doc_txt"
    output_dir = Path(args.output_dir).resolve()

    assignments = load_cluster_assignments(clustering_dir, args.method)
    log.info("Loaded %d assignments", len(assignments))

    cluster_texts = group_texts_by_cluster(assignments, doc_txt_dir)
    log.info("Found %d clusters", len(cluster_texts))

    labels = generate_labels(cluster_texts, top_n=args.top_n)
    labels = add_tfidf_terms(labels, cluster_texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.method}_cluster_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2, ensure_ascii=False)
    log.info("Saved → %s", out_path)

    print("\n" + "="*60)
    print("CLUSTER LABELS SUMMARY")
    print("="*60)
    for cid in sorted(labels.keys()):
        info = labels[cid]
        print(f"\nCluster {cid} ({info['num_patents']} patents)")
        print(f"  Label    : {info['label']}")
        print(f"  Keywords : {', '.join(info['keywords'])}")
        if info.get('tfidf_terms'):
            print(f"  TF-IDF   : {', '.join(info['tfidf_terms'])}")
    print("="*60)

if __name__ == "__main__":
    main()
