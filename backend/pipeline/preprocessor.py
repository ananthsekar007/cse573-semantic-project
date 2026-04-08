"""
Patent Preprocessing Pipeline
================================
Reads raw JSON files produced by scraper.py and outputs cleaned, 
normalized text files ready for the embedding pipeline.

For each patent it produces:
  - <output_dir>/cleaned_json/<patent_id>.json  — structured cleaned fields
  - <output_dir>/cleaned_txt/<patent_id>.txt    — flat cleaned text for embedding

Cleaning steps applied:
  1. Language detection  — skip non-English patents
  2. Boilerplate removal — cross-references, legal headers, figure callouts
  3. HTML/XML artifact   — residual tags, encoded entities
  4. Whitespace norm     — collapse tabs, newlines, excess spaces
  5. Lowercasing
  6. Tokenization
  7. Stopword removal
  8. Lemmatization
  9. Catalog update      — writes cleaned_catalog.json with per-patent stats

Usage:
    python preprocessor.py                          # uses ./data/raw and ./data/processed
    python preprocessor.py --input-dir ./data/raw --output-dir ./data/processed
    python preprocessor.py --dry-run                # print stats without writing files
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import nltk
import spacy
from langdetect import detect, LangDetectException

# ---------------------------------------------------------------------------
# NLTK / spaCy setup — download required resources on first run
# ---------------------------------------------------------------------------

def _ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

_ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# spaCy English model for lemmatization
# Install with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

ENGLISH_STOPWORDS = set(stopwords.words("english"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boilerplate patterns — compiled once at module load
# ---------------------------------------------------------------------------

# Cross-reference block: "This application is a continuation of..."
# Typically occupies the entire first paragraph of the description
_RE_CROSS_REF = re.compile(
    r"(this application (is a continuation|claims priority|hereby incorporates)[^.]*\.[\s]*)+",
    re.IGNORECASE | re.DOTALL,
)

# Legal incorporation by reference lines
_RE_INCORPORATION = re.compile(
    r"(each of which is hereby incorporated by reference[^.]*\.)",
    re.IGNORECASE,
)

# Patent number citations: "U.S. Pat. No. 8,879,416" / "U.S. application Ser. No. 17/117,107"
_RE_PATENT_CITES = re.compile(
    r"(U\.S\.|WO|EP|PCT)\s+(Pat\.|patent|application)\s+(App\.|Pub\.|No\.)\s+[\w,/]+",
    re.IGNORECASE,
)

# Figure references: "FIG. 1", "FIG. 14", "FIGS. 1-3"
_RE_FIG_REFS = re.compile(r"\bfigs?\.?\s*\d+[\w\-]*\b", re.IGNORECASE)

# Paragraph numbers: "[0001]", "[0042]"
_RE_PARA_NUMS = re.compile(r"\[\d{4}\]")

# Residual HTML / XML tags
_RE_HTML_TAGS = re.compile(r"<[^>]+>")

# HTML entities: &hellip; &amp; etc.
_RE_HTML_ENTITIES = re.compile(r"&[a-zA-Z]+;|&#\d+;")

# Encoded unicode artifacts from bad extraction: â€™ Â§ etc.
_RE_ENCODING_ARTIFACTS = re.compile(r"[âÂ©®°±×÷€£¥§¶†‡•…‰›‹«»]")

# Claim preamble boilerplate (repeated in every patent)
_RE_CLAIM_PREAMBLE = re.compile(
    r"(a non-transitory computer.readable medium containing instructions[^,]*,\s*"
    r"when executed[^,]*,\s*cause[^:]*:)",
    re.IGNORECASE | re.DOTALL,
)

# Attorney docket numbers: "PWS-71700US01"
_RE_DOCKET = re.compile(r"\b[A-Z]{2,6}-\d{4,}[A-Z]{0,4}\d{0,2}\b")

# Excess whitespace
_RE_WHITESPACE = re.compile(r"[ \t]{2,}")
_RE_NEWLINES   = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Data model for cleaned patent
# ---------------------------------------------------------------------------

@dataclass
class CleanedPatent:
    patent_id:          str
    publication_number: str
    title:              str
    domain:             str
    # Raw cleaned text (boilerplate removed, entities fixed, but NOT lemmatized)
    # Used for RAG chunking — preserves readability
    abstract_clean:     str       = ""
    claims_clean:       list[str] = field(default_factory=list)
    description_clean:  str       = ""
    # Fully processed text (lowercased, stopwords removed, lemmatized)
    # Used for document-level embedding
    abstract_processed: str       = ""
    claims_processed:   str       = ""
    description_processed: str    = ""
    # Metadata passthrough
    inventor:           str       = ""
    assignee:           str       = ""
    filing_date:        str       = ""
    grant_date:         str       = ""
    publication_date:   str       = ""
    language_detected:  str       = ""
    token_count_raw:    int       = 0
    token_count_clean:  int       = 0

    def to_rag_txt(self) -> str:
        """
        Human-readable cleaned text for RAG chunking.
        Preserves sentence structure and casing — fed into chunk-level embeddings.
        """
        claims_text = "\n".join(f"  {c}" for c in self.claims_clean) if self.claims_clean else "  N/A"
        return f"""PATENT ID: {self.publication_number}
TITLE: {self.title}
DOMAIN: {self.domain}
INVENTOR: {self.inventor}
ASSIGNEE: {self.assignee}
FILING DATE: {self.filing_date}

=== ABSTRACT ===
{self.abstract_clean or 'N/A'}

=== CLAIMS ===
{claims_text}

=== DESCRIPTION ===
{self.description_clean or 'N/A'}
""".strip()

    def to_doc_embedding_txt(self) -> str:
        """
        Input text for document-level SBERT embedding (used for clustering).

        Uses clean but NOT NLP-processed text because:
        - SBERT is a transformer trained on natural language — it performs
          better with grammatically coherent input
        - Stopword removal hurts SBERT (words like "not", "without" carry
          semantic meaning in patent claims)
        - SBERT hard-truncates at 512 tokens so we want the highest-signal
          content first: title + abstract + first independent claim only

        The _processed fields (lemmatized, stopwords removed) are reserved
        for KeyBERT cluster labeling and TF-IDF baseline experiments.
        """
        first_claim = self.claims_clean[0] if self.claims_clean else ""
        return f"{self.title}. {self.abstract_clean} {first_claim}".strip()


# ---------------------------------------------------------------------------
# Step 1 — Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Returns ISO 639-1 language code (e.g. 'en', 'zh-cn', 'de') or 'unknown'.
    Samples the first 800 chars for speed — sufficient for language detection.
    """
    sample = text[:800].strip()
    if not sample:
        return "unknown"
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# Step 2 — Boilerplate and artifact removal
# ---------------------------------------------------------------------------

def remove_boilerplate(text: str) -> str:
    """
    Removes patent-specific boilerplate that adds no semantic value:
    cross-references, figure labels, paragraph numbers, legal preambles.
    Operates on raw text before tokenization.
    """
    text = _RE_CROSS_REF.sub(" ", text)
    text = _RE_INCORPORATION.sub(" ", text)
    text = _RE_PATENT_CITES.sub(" ", text)
    text = _RE_FIG_REFS.sub(" ", text)
    text = _RE_PARA_NUMS.sub(" ", text)
    text = _RE_HTML_TAGS.sub(" ", text)
    text = _RE_HTML_ENTITIES.sub(" ", text)
    text = _RE_ENCODING_ARTIFACTS.sub(" ", text)
    text = _RE_CLAIM_PREAMBLE.sub(" ", text)
    text = _RE_DOCKET.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_NEWLINES.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Boilerplate removal + whitespace normalization. Preserves casing."""
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Step 3 — NLP processing (lowercase → tokenize → stopwords → lemmatize)
# ---------------------------------------------------------------------------

def process_text(text: str) -> str:
    """
    Full NLP pipeline:
      lowercase → tokenize → remove stopwords + punctuation → lemmatize
    Returns a space-joined string of lemmas.
    Uses spaCy for lemmatization (more accurate than NLTK WordNetLemmatizer
    for technical/domain-specific vocabulary).
    """
    if not text.strip():
        return ""

    text = text.lower()

    # spaCy processes in chunks to stay within its token limit
    chunk_size  = 100_000
    all_lemmas  = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        doc   = nlp(chunk)
        lemmas = [
            token.lemma_
            for token in doc
            if (
                token.is_alpha                          # letters only
                and not token.is_stop                   # spaCy stopwords
                and token.text not in ENGLISH_STOPWORDS # NLTK stopwords
                and len(token.text) > 2                 # drop very short tokens
            )
        ]
        all_lemmas.extend(lemmas)

    return " ".join(all_lemmas)


# ---------------------------------------------------------------------------
# Main per-patent processing function
# ---------------------------------------------------------------------------

def process_patent(raw: dict) -> Optional[CleanedPatent]:
    """
    Takes a raw patent dict (loaded from scraper JSON) and returns
    a CleanedPatent or None if the patent should be skipped.
    """
    patent_id  = raw.get("patent_id", "")
    pub_number = raw.get("publication_number", "")
    title      = raw.get("title", "")
    domain     = raw.get("domain", "")
    abstract   = raw.get("abstract", "")
    claims     = raw.get("claims", [])
    description = raw.get("description", "")

    # --- Step 1: Language detection ---
    # Sample title + abstract + first 500 chars of description for detection
    sample_text = f"{title} {abstract} {description[:500]}"
    lang = detect_language(sample_text)

    log.info("    Language detected: %s", lang)

    if lang != "en":
        log.warning("    SKIP — non-English patent (lang=%s): %s", lang, patent_id)
        return None

    # --- Step 2: Clean each section ---
    abstract_clean    = clean_text(abstract)
    description_clean = clean_text(description)
    claims_clean      = [clean_text(c) for c in claims if isinstance(c, str) and c.strip()]

    # Raw token count (before NLP processing) — useful for stats
    raw_combined  = f"{abstract} {' '.join(claims)} {description}"
    token_count_raw = len(word_tokenize(raw_combined))

    # --- Step 3: NLP processing for document-level embedding ---
    abstract_processed    = process_text(abstract_clean)
    description_processed = process_text(description_clean)
    # For claims, join all cleaned claims then process as one block
    claims_combined_clean = " ".join(claims_clean)
    claims_processed      = process_text(claims_combined_clean)

    clean_combined = f"{abstract_clean} {claims_combined_clean} {description_clean}"
    token_count_clean = len(word_tokenize(clean_combined))

    return CleanedPatent(
        patent_id             = patent_id,
        publication_number    = pub_number,
        title                 = title,
        domain                = domain,
        abstract_clean        = abstract_clean,
        claims_clean          = claims_clean,
        description_clean     = description_clean,
        abstract_processed    = abstract_processed,
        claims_processed      = claims_processed,
        description_processed = description_processed,
        inventor              = raw.get("inventor", ""),
        assignee              = raw.get("assignee", ""),
        filing_date           = raw.get("filing_date", ""),
        grant_date            = raw.get("grant_date", ""),
        publication_date      = raw.get("publication_date", ""),
        language_detected     = lang,
        token_count_raw       = token_count_raw,
        token_count_clean     = token_count_clean,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_cleaned_patent(patent: CleanedPatent, output_dir: Path) -> None:
    """
    Writes:
      <output_dir>/cleaned_json/<patent_id>.json  — full structured cleaned data
      <output_dir>/rag_txt/<patent_id>.txt         — readable text for RAG chunking
      <output_dir>/doc_txt/<patent_id>.txt         — processed text for doc embedding
    """
    (output_dir / "cleaned_json").mkdir(parents=True, exist_ok=True)
    (output_dir / "rag_txt").mkdir(parents=True, exist_ok=True)
    (output_dir / "doc_txt").mkdir(parents=True, exist_ok=True)

    with open(output_dir / "cleaned_json" / f"{patent.patent_id}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(patent), f, indent=2, ensure_ascii=False)

    with open(output_dir / "rag_txt" / f"{patent.patent_id}.txt", "w", encoding="utf-8") as f:
        f.write(patent.to_rag_txt())

    with open(output_dir / "doc_txt" / f"{patent.patent_id}.txt", "w", encoding="utf-8") as f:
        f.write(patent.to_doc_embedding_txt())


def save_cleaned_catalog(patents: list[CleanedPatent], output_dir: Path) -> None:
    catalog = [
        {
            "patent_id":          p.patent_id,
            "publication_number": p.publication_number,
            "title":              p.title,
            "domain":             p.domain,
            "language_detected":  p.language_detected,
            "token_count_raw":    p.token_count_raw,
            "token_count_clean":  p.token_count_clean,
            "has_abstract":       bool(p.abstract_clean),
            "has_claims":         bool(p.claims_clean),
            "has_description":    bool(p.description_clean),
            "claim_count":        len(p.claims_clean),
        }
        for p in patents
    ]
    path = output_dir / "cleaned_catalog.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    log.info("Cleaned catalog saved → %s (%d entries)", path, len(catalog))


def load_cached_ids(output_dir: Path) -> set[str]:
    json_dir = output_dir / "cleaned_json"
    if not json_dir.exists():
        return set()
    return {p.stem for p in json_dir.glob("*.json")}


# ---------------------------------------------------------------------------
# Parallel worker — must be a top-level function for ProcessPoolExecutor
# pickling. Each worker process loads spaCy once and handles one file.
# ---------------------------------------------------------------------------

def _process_file(path: Path) -> tuple[str, Optional[dict], str]:
    """
    Worker function executed in a subprocess.
    Returns (patent_id, cleaned_dict_or_None, skip_reason)
    skip_reason is one of: "ok" | "empty" | "language" | "failed"
    """
    patent_id = path.stem
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        return patent_id, None, "failed"

    if not raw.get("abstract") and not raw.get("description"):
        return patent_id, None, "empty"

    cleaned = process_patent(raw)
    if cleaned is None:
        return patent_id, None, "language"

    return patent_id, asdict(cleaned), "ok"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    input_dir:  Path,
    output_dir: Path,
    dry_run:    bool = False,
    workers:    int  = max(1, (os.cpu_count() or 2) - 1),
) -> None:
    """
    Parallel preprocessing pipeline.

    Each file is processed in a separate worker process so spaCy
    lemmatization runs truly concurrently across CPU cores.
    Workers is capped at cpu_count - 1 to keep the main process
    responsive for I/O and logging.
    """
    raw_json_dir = input_dir / "json"

    if not raw_json_dir.exists():
        raise FileNotFoundError(f"No json/ directory found in {input_dir}. Run scraper.py first.")

    raw_files  = sorted(raw_json_dir.glob("*.json"))
    cached_ids = load_cached_ids(output_dir)

    # Filter out already-processed files before submitting to workers
    pending = [p for p in raw_files if p.stem not in cached_ids]
    n_cached = len(raw_files) - len(pending)

    log.info("Found %d raw patents. %d cached, %d to process. Workers: %d",
             len(raw_files), n_cached, len(pending), workers)

    all_cleaned: list[CleanedPatent] = []
    stats = {
        "processed":        0,
        "skipped_cache":    n_cached,
        "skipped_language": 0,
        "skipped_empty":    0,
        "failed":           0,
    }

    if not pending:
        log.info("Nothing to do — all patents already cleaned.")
        return

    if dry_run:
        log.info("[DRY RUN] Would process %d patents with %d workers.", len(pending), workers)
        return

    # Ensure output directories exist before workers start writing
    (output_dir / "cleaned_json").mkdir(parents=True, exist_ok=True)
    (output_dir / "rag_txt").mkdir(parents=True, exist_ok=True)
    (output_dir / "doc_txt").mkdir(parents=True, exist_ok=True)

    completed = 0
    total     = len(pending)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_file, path): path for path in pending}

        for future in as_completed(futures):
            path      = futures[future]
            completed += 1

            try:
                patent_id, cleaned_dict, reason = future.result()
            except Exception as e:
                log.error("[%d/%d] Worker exception for %s: %s", completed, total, path.stem, e)
                stats["failed"] += 1
                continue

            if reason == "failed":
                log.error("[%d/%d] FAILED to load: %s", completed, total, patent_id)
                stats["failed"] += 1

            elif reason == "empty":
                log.warning("[%d/%d] SKIP (empty): %s", completed, total, patent_id)
                stats["skipped_empty"] += 1

            elif reason == "language":
                log.info("[%d/%d] SKIP (non-English): %s", completed, total, patent_id)
                stats["skipped_language"] += 1

            elif reason == "ok":
                # Reconstruct dataclass from dict returned by worker process
                cleaned = CleanedPatent(**cleaned_dict)
                save_cleaned_patent(cleaned, output_dir)
                all_cleaned.append(cleaned)
                stats["processed"] += 1

                log.info("[%d/%d] OK  %s  tokens: %d→%d  claims: %d",
                         completed, total, patent_id,
                         cleaned.token_count_raw,
                         cleaned.token_count_clean,
                         len(cleaned.claims_clean))

    if all_cleaned:
        save_cleaned_catalog(all_cleaned, output_dir)

    log.info("")
    log.info("━" * 60)
    log.info("DONE")
    log.info("  Processed       : %d", stats["processed"])
    log.info("  Cache hits      : %d", stats["skipped_cache"])
    log.info("  Skipped (lang)  : %d", stats["skipped_language"])
    log.info("  Skipped (empty) : %d", stats["skipped_empty"])
    log.info("  Failed          : %d", stats["failed"])
    log.info("Output → %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patent preprocessing pipeline")
    parser.add_argument("--input-dir",  default="../data/raw",       help="Scraper output directory")
    parser.add_argument("--output-dir", default="../data/processed", help="Cleaned output directory")
    parser.add_argument("--dry-run",    action="store_true",         help="Stats only, no files written")
    parser.add_argument("--workers",    type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of parallel worker processes (default: cpu_count - 1)")
    args = parser.parse_args()

    run_pipeline(
        input_dir  = Path(args.input_dir),
        output_dir = Path(args.output_dir),
        dry_run    = args.dry_run,
        workers    = args.workers,
    )