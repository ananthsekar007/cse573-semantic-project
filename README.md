# Semantic Patent Analysis and Retrieval System

**CSE 573 · Spring 2026 · Project 9 · Group 21**

> An end-to-end NLP pipeline for clustering U.S. patents by technology domain,
> enabling conversational search via RAG, and scoring novelty of new patent proposals
> against a curated prior-art corpus.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1 — Data Collection](#stage-1--data-collection)
  - [Stage 2 — Preprocessing](#stage-2--preprocessing)
  - [Stage 3 — Clustering](#stage-3--clustering-baseline--sota)
- [Evaluation and Results](#evaluation-and-results)
- [Novelty and Intelligence Layer](#novelty-and-intelligence-layer)
- [Data Flow](#data-flow)
- [Configuration](#configuration)
- [Current Status](#current-status)

---

## Project Overview

The exponential growth of patent filings has made it increasingly difficult to
analyze technological trends, identify domain groupings, and assess novelty
against prior art. Traditional keyword-based search methods fail to capture
deeper semantic similarities between patents.

This system addresses that by building a pipeline that:

- **Collects** full patent text from Google Patents via SerpAPI across 5 technology domains
- **Cleans and normalizes** raw patent text for semantic analysis
- **Embeds** patents as dense vectors using Sentence-BERT (planned)
- **Clusters** patents into technology domains using HDBSCAN + UMAP (planned)
- **Enables conversational search** via a RAG pipeline backed by FAISS + Grok LLM (planned)
- **Scores novelty** of new patent proposals against prior art using cosine similarity

### Target Domains

| Domain Key | Description |
|---|---|
| `ai_machine_learning` | Deep learning architectures, model optimization, inference systems |
| `biotechnology` | CRISPR, mRNA delivery, gene-editing platforms |
| `semiconductor` | Advanced node processes, chip architectures |
| `telecommunications_5g` | Millimeter wave, MIMO, network slicing |
| `renewable_energy` | Solar, wind, and energy storage technologies |

### Dataset

- **Source:** Google Patents via SerpAPI (`google_patents` + `google_patents_details` engines)
- **Scope:** U.S.-issued English-language patents
- **Size:** 100 patents per domain (500 total for full run; 10 per domain recommended for development)
- **Split:** 300 train / 200 test (60/40) for embedding and validation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE                      │
│                                                          │
│  SerpAPI ──► Preprocessing ──► SBERT Embeddings          │
│                                    │                     │
│                          ┌─────────┴──────────┐         │
│                          ▼                    ▼         │
│                   UMAP + HDBSCAN         FAISS Index     │
│                   (doc-level)            (chunk-level)   │
│                          │                    │         │
│                   KeyBERT Labels         RAG Pipeline    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    ONLINE PIPELINE                       │
│                                                          │
│  React UI ──► FastAPI ──► FAISS Retrieval ──► Grok LLM  │
│                                                          │
│  Outputs:                                                │
│    • UMAP cluster dashboard                              │
│    • Conversational patent Q&A                           │
│    • Novelty similarity score                            │
└─────────────────────────────────────────────────────────┘
```

### Two FAISS Indexes

The system maintains two separate FAISS indexes with different embedding strategies:

| Index | Embedding Unit | Purpose |
|---|---|---|
| `faiss_doc.index` | Whole patent: title + abstract + claim 1 | HDBSCAN clustering, UMAP visualization, novelty scoring |
| `faiss_chunk.index` | 300–500 token overlapping chunks | RAG retrieval — answers specific user queries |

Document-level embeddings use clean but **unprocessed** text (SBERT performs better
on grammatically coherent input). Chunk-level embeddings use the full cleaned
description split into overlapping windows.

---

## Technology Stack

| Component | Technology |
|---|---|
| Data collection | SerpAPI SDK (`serpapi`) |
| HTML parsing | BeautifulSoup4 + lxml |
| Language detection | langdetect |
| NLP preprocessing | spaCy `en_core_web_sm`, NLTK |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (768-dim) |
| Vector store | FAISS (two indexes) |
| Clustering | HDBSCAN |
| Dimensionality reduction | UMAP |
| Cluster labeling | KeyBERT + TF-IDF |
| LLM | Grok (xAI) via OpenAI-compatible API |
| Backend | FastAPI (Python) |
| Frontend | React + Vite |

---

## Project Structure

```
patent-analysis/
│
├── pipeline/
│   ├── scraper.py           # Stage 1 — data collection from SerpAPI
│   ├── preprocessor.py      # Stage 2 — cleaning, language detection, NLP
│   └── requirements.txt     # Python dependencies for pipeline stages
│
├── backend/                 # (planned) FastAPI backend
│   └── app/
│       ├── main.py
│       ├── api/routes/
│       │   ├── patents.py
│       │   ├── chat.py
│       │   ├── novelty.py
│       │   └── clusters.py
│       └── core/
│           ├── embedder.py
│           ├── faiss_store.py
│           ├── clusterer.py
│           ├── labeler.py
│           └── grok_client.py
│
├── frontend/                # (planned) React frontend
│   └── src/
│       ├── components/
│       │   ├── ClusterMap.jsx
│       │   ├── ChatPanel.jsx
│       │   └── NoveltyChecker.jsx
│       └── App.jsx
│
└── data/
    ├── raw/
    │   ├── json/            # Raw patent JSON from scraper
    │   ├── txt/             # Flat text from scraper (title+abstract+claims+description)
    │   └── catalog.json     # Index of all collected patents
    └── processed/
        ├── cleaned_json/    # Cleaned structured patent data
        ├── rag_txt/         # Readable cleaned text for RAG chunking
        ├── doc_txt/         # title + abstract + claim 1 for doc-level embedding
        └── cleaned_catalog.json
```

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- A SerpAPI account and API key ([serpapi.com](https://serpapi.com))
- A Grok API key from xAI ([x.ai](https://x.ai)) — required for RAG stage (not yet implemented)

### 1. Clone the repository

```bash
git clone <repo-url>
cd patent-analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Set environment variables

```bash
export SERPAPI_KEY=your_serpapi_key_here
```

Or create a `.env` file in the project root:

```
SERPAPI_KEY=your_serpapi_key_here
GROK_API_KEY=your_grok_key_here     # needed for RAG stage later
```

---

## Pipeline Stages

### Stage 1 — Data Collection

**Script:** `pipeline/scraper.py`

Collects patent metadata and full text from Google Patents via SerpAPI and
stores each patent as structured JSON + flat text.

#### What it does

1. Queries `google_patents` engine with domain-specific search terms (100 results per domain)
2. Filters out scholar results and non-English patents
3. Fetches full details per patent via `google_patents_details` engine (abstract, claims)
4. Scrapes the description HTML from `description_link` using BeautifulSoup
5. Saves each patent as:
   - `data/raw/json/<patent_id>.json` — structured source of truth
   - `data/raw/txt/<patent_id>.txt` — flat text with delimited sections
6. Writes `data/raw/catalog.json` — summary of the full collected corpus

#### Output format (`data/raw/txt/<patent_id>.txt`)

```
PATENT ID: US20240188189A1
TITLE: 5G OpenRAN Controller
DOMAIN: telecommunications_5g
...

=== ABSTRACT ===
A device, method and software are presented for a 5G OpenRAN controller...

=== CLAIMS ===
  1. A 5G OpenRAN controller, comprising:...

=== DESCRIPTION ===
[BACKGROUND]
5G is the next generation Mobile Communication technology...
```

#### Usage

```bash
# Recommended: dry run first to verify patent counts before spending API credits
python pipeline/scraper.py --dry-run

# Full collection run
python pipeline/scraper.py

# Custom paths
python pipeline/scraper.py --api-key YOUR_KEY --output-dir ./data/raw
```

#### API credit usage

| Operation | Calls |
|---|---|
| Phase 1 — metadata collection | 5 (one per domain) |
| Phase 2 — detail fetching | up to 500 (one per patent) |
| Phase 3 — description HTML | free (direct HTTP, not SerpAPI) |
| **Total** | **~505 credits for full run** |

> **Tip:** Use `PATENTS_PER_DOMAIN = 10` in `scraper.py` during development to
> limit spend to ~55 credits while testing the full pipeline end-to-end.

---

### Stage 2 — Preprocessing

**Script:** `pipeline/preprocessor.py`

Reads raw JSON from Stage 1 and produces cleaned, normalized text ready for
the embedding pipeline. Runs in parallel using `ProcessPoolExecutor`.

#### What it does

1. **Language detection** — detects language of each patent using `langdetect`;
   skips non-English patents and logs them to stats
2. **Boilerplate removal** — strips cross-references, patent number citations,
   figure references, paragraph numbers, HTML artifacts, encoding noise, and
   legal claim preambles using compiled regex patterns
3. **Whitespace normalization** — collapses excess spaces, tabs, and blank lines
4. **NLP processing** (for KeyBERT/TF-IDF use) — lowercasing, tokenization,
   stopword removal (NLTK + spaCy), lemmatization (spaCy `en_core_web_sm`)
5. **Output generation** — writes three files per patent plus a cleaned catalog

#### Why two output text formats

| File | Content | Used for |
|---|---|---|
| `rag_txt/<id>.txt` | Full cleaned text, casing preserved, readable | RAG chunk-level embeddings |
| `doc_txt/<id>.txt` | Title + abstract_clean + claim 1 only | Document-level SBERT embeddings for clustering |

The `doc_txt` intentionally uses **clean but not lemmatized** text. SBERT is a
transformer trained on natural language — it performs better with grammatically
coherent input. Stopword removal hurts SBERT because words like "not" and
"without" carry semantic meaning in patent claims. The fully lemmatized
`*_processed` fields in the JSON are reserved for KeyBERT cluster labeling.

The description is excluded from `doc_txt` because SBERT hard-truncates at 512
tokens. Patent descriptions are 15,000–50,000 tokens — feeding them in means
SBERT would only ever see the background/prior art section (the least
distinctive part). The abstract is purpose-built to summarize the invention
concisely and fits comfortably within the token limit.

#### Parallel processing

The pipeline uses `ProcessPoolExecutor` with `cpu_count - 1` workers by default.
Each worker process loads spaCy independently, so lemmatization runs truly
concurrently across all available CPU cores.

```bash
# Default (cpu_count - 1 workers)
python pipeline/preprocessor.py

# Explicit worker count
python pipeline/preprocessor.py --workers 4

# Single process — easier for debugging
python pipeline/preprocessor.py --workers 1

# Dry run — shows how many patents would be processed
python pipeline/preprocessor.py --dry-run
```

#### Resumability

Both stages are fully resumable. Re-running either script skips patents that
have already been processed by checking for existing output files. Safe to
interrupt and restart at any point.

---

### Stage 3 — Clustering (Baseline + SOTA)

**Script:** `pipeline/clusterer.py`

Runs two algorithm tracks for technology-domain discovery:

1. **Baseline (implemented): TF-IDF + K-Means**
2. **SOTA track (implemented): SBERT doc embeddings + UMAP + HDBSCAN**

#### Baseline approach (recommended for your project milestone)

- Input: `data/processed/doc_txt/<id>.txt`
- Vectorization: TF-IDF or BM25 with unigram+bigram features
- Clustering: K-Means (`k=5` by default for the 5 target domains)
- Optional linear projection: PCA before K-Means (`--pca-components > 0`)
- Outputs:
  - `data/clustering/baseline_cluster_assignments.json`
  - `data/clustering/baseline_metrics.json`
- Metrics generated:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Homogeneity
  - Silhouette score
  - Cluster purity

#### SOTA track

- Input: `data/embeddings/doc_embeddings.npy`
- Projection: UMAP
- Clustering: HDBSCAN (noise-aware, no fixed k)
- Outputs:
  - `data/clustering/sota_cluster_assignments.json`
  - `data/clustering/sota_metrics.json`

#### Usage

```bash
# Baseline only
python pipeline/clusterer.py --method baseline

# Baseline with BM25 vectors
python pipeline/clusterer.py --method baseline --vectorizer bm25

# Baseline with PCA linear projection before K-Means
python pipeline/clusterer.py --method baseline --vectorizer tfidf --pca-components 50

# SOTA only (requires embeddings from embedder.py)
python pipeline/clusterer.py --method sota

# Run both tracks
python pipeline/clusterer.py --method both
```

### Keyword retrieval baselines (inverted index + Boolean syntax)

**Script:** `pipeline/keyword_search.py`

Implements exact-term lookup with an inverted index and supports Boolean query
syntax (`AND`, `OR`, `NOT`, and parentheses).

```bash
# Build index from processed corpus
python pipeline/keyword_search.py --build-index

# Exact/Boolean keyword search
python pipeline/keyword_search.py --query "semiconductor AND lithography"
python pipeline/keyword_search.py --query "(5g OR telecom) AND NOT satellite"
```

---

## Evaluation and Results

The clustering stage is evaluated against the known patent domain labels that are
already present in the cleaned corpus metadata. The baseline is the primary
milestone deliverable because it is deterministic, easy to reproduce, and gives a
clear comparison point against the more advanced HDBSCAN track.

### Evaluation protocol

1. Run preprocessing so `data/processed/doc_txt/<id>.txt` and
  `data/processed/cleaned_json/<id>.json` exist for the same patent IDs.
2. Run `pipeline/clusterer.py --method baseline` to assign each patent to a
  K-Means cluster.
3. Compare predicted clusters against the domain field stored in the cleaned
  JSON metadata.
4. Save the resulting assignments and summary metrics under
  `data/clustering/` for reporting and analysis.

### Metrics reported

| Metric | Purpose |
|---|---|
| ARI | Measures agreement with the known domain labels after accounting for chance |
| NMI | Captures shared information between predicted clusters and true domains |
| Homogeneity | Checks whether each cluster contains patents from a single domain |
| Silhouette score | Measures how well-separated the TF-IDF clusters are |
| Cluster purity | Fraction of patents assigned to the dominant label in each cluster |

### Result artifacts

The baseline run writes two files:

- `data/clustering/baseline_cluster_assignments.json`
- `data/clustering/baseline_metrics.json`

The SOTA run writes the equivalent `sota_*` files. The metrics file is the one to
reference in the presentation because it captures the quantitative evaluation of
the algorithmic baseline. If the corpus has not been collected yet, the code is
still ready, but the numeric results will only appear once the processed dataset
is present.

### Novelty and intelligence layer

**Script:** `pipeline/novelty.py`

The novelty baseline uses the FAISS document index to compare a patent against
its nearest prior-art neighbor. The novelty score is defined as `1 - max_similarity`.

#### What it does

1. Loads `data/embeddings/doc_embeddings.npy` and `data/embeddings/doc_metadata.json`
2. Builds or loads `data/embeddings/faiss_doc.index`
3. Scores each patent against its nearest non-self neighbor
4. Writes a per-patent score file and a summary statistics file

#### Outputs

- `data/novelty/novelty_scores.json`
- `data/novelty/novelty_summary.json`

#### Usage

```bash
# Corpus-level novelty scoring
python pipeline/novelty.py

# Score a new candidate patent text
python pipeline/novelty.py --query-text "A device for..."

# Score a text file
python pipeline/novelty.py --query-file ./candidate_patent.txt
```

---

## Data Flow

```
SerpAPI
  │
  ▼
scraper.py
  │
  ├── data/raw/json/<id>.json       ← structured raw data
  ├── data/raw/txt/<id>.txt         ← flat text (scraper output)
  └── data/raw/catalog.json
  │
  ▼
preprocessor.py
  │
  ├── data/processed/cleaned_json/<id>.json   ← full cleaned structured data
  ├── data/processed/rag_txt/<id>.txt         ← readable text for RAG chunking
  ├── data/processed/doc_txt/<id>.txt         ← title+abstract+claim1 for SBERT
  └── data/processed/cleaned_catalog.json
  │
  ▼
  (embedding pipeline — coming next)
```

---

## Configuration

Key constants are defined at the top of each script and can be modified without
touching pipeline logic.

### `scraper.py`

| Constant | Default | Description |
|---|---|---|
| `DOMAINS` | 5 domains | Search queries per technology area |
| `PATENTS_PER_DOMAIN` | `100` | Set to `10` for development |
| `DESCRIPTION_MAX_CHARS` | `100_000` | Truncation limit for description text |
| `REQUEST_DELAY` | `1.2` | Seconds between SerpAPI SDK calls |
| `MAX_RETRIES` | `3` | Retry attempts on API failure |

### `preprocessor.py`

| Constant | Default | Description |
|---|---|---|
| `workers` | `cpu_count - 1` | Parallel worker processes |

---

## Current Status

| Stage | Status | Script |
|---|---|---|
| Data collection | ✅ Complete | `scraper.py` |
| Preprocessing | ✅ Complete | `preprocessor.py` |
| Embedding generation & Fiass Store | ✅ Complete | `embedder.py` |
| Clustering and evaluation | ✅ Baseline + SOTA implemented | `clusterer.py` |
| KeyBERT labeling | 🔲 Planned | `labeler.py` (planned) |
| RAG pipeline | 🔲 Planned | `chat.py` (planned) |
| Novelty scoring | ✅ Baseline implemented | `novelty.py` |
| FastAPI backend | 🔲 Planned | `backend/` (planned) |
| React frontend | 🔲 Planned | `frontend/` (planned) |