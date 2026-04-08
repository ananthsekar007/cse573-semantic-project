"""
Patent Data Ingestion Pipeline
================================
Phase 1: Collect patent IDs + metadata via google_patents engine (100 per page)
Phase 2: Fetch full details per patent via google_patents_details engine
Phase 3: Scrape description HTML from description_link
Phase 4: Store as structured JSON + flat .txt per patent

Usage:
    python scraper.py                        # uses SERPAPI_KEY env var
    python scraper.py --dry-run              # collect IDs only, skip detail fetching
    python scraper.py --output-dir ./data    # custom output directory
"""

import os
import re
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import serpapi
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOMAINS = {
    "ai_machine_learning":   "artificial intelligence machine learning deep learning",
    "biotechnology":         "CRISPR gene therapy mRNA biotechnology",
    "semiconductor":         "semiconductor chip design advanced node fabrication",
    "telecommunications_5g": "5G millimeter wave MIMO network slicing",
    "renewable_energy":      "solar panel wind turbine energy storage battery",
}

PATENTS_PER_DOMAIN    = 10   # exactly one page at num=100

DESCRIPTION_MAX_CHARS = 100_000  # patent descriptions can be very long; only truncate extreme cases
REQUEST_DELAY         = 1.2   # seconds between SDK calls
MAX_RETRIES           = 3
BACKOFF_FACTOR        = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Patent:
    patent_id:          str
    publication_number: str
    title:              str
    domain:             str
    abstract:           str       = ""
    claims:             list[str] = field(default_factory=list)
    description:        str       = ""
    snippet:            str       = ""
    inventor:           str       = ""
    assignee:           str       = ""
    filing_date:        str       = ""
    grant_date:         str       = ""
    publication_date:   str       = ""
    priority_date:      str       = ""
    patent_link:        str       = ""
    language:           str       = "en"
    country_status:     dict      = field(default_factory=dict)

    def to_txt(self) -> str:
        claims_text = "\n".join(f"  {c}" for c in self.claims) if self.claims else "  N/A"
        return f"""PATENT ID: {self.publication_number}
TITLE: {self.title}
DOMAIN: {self.domain}
INVENTOR: {self.inventor}
ASSIGNEE: {self.assignee}
FILING DATE: {self.filing_date}
GRANT DATE: {self.grant_date}

=== ABSTRACT ===
{self.abstract or 'N/A'}

=== CLAIMS ===
{claims_text}

=== DESCRIPTION ===
{self.description or 'N/A'}
""".strip()


# ---------------------------------------------------------------------------
# SerpAPI SDK helpers
# ---------------------------------------------------------------------------

def _sdk_search(client: serpapi.Client, params: dict) -> Optional[dict]:
    """
    Wraps client.search() with retry + exponential backoff.
    Returns the raw results dict or None on persistent failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            results = client.search(params)
            return dict(results)
        except Exception as e:
            wait = REQUEST_DELAY * (BACKOFF_FACTOR ** attempt)
            log.warning("SDK error (attempt %d/%d): %s — retrying in %.1fs", attempt, MAX_RETRIES, e, wait)
            time.sleep(wait)

    log.error("All %d retries failed for params: %s", MAX_RETRIES, params)
    return None


def _scrape_description(url: str) -> str:
    """
    Fetches the SerpAPI description HTML page and extracts clean patent text.

    The SerpAPI description page structure for Google Patents looks like:
        <body>
          <ul class="description" mxw-id="...">
            <heading id="h-0001">CROSS-REFERENCE ...</heading>
            <li><div class="description-line">...</div></li>
            ...
          </ul>
        </body>

    We target <ul class="description"> directly, then extract text from
    every <div class="description-line"> and <heading> in document order,
    preserving section headings so the output is readable.
    """
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        log.warning("Failed to fetch description HTML: %s", e)
        return ""

    soup = BeautifulSoup(resp.text, "lxml")

    # ------------------------------------------------------------------ #
    # Primary target: <ul class="description"> — the canonical container  #
    # in SerpAPI-served Google Patents description pages                  #
    # ------------------------------------------------------------------ #
    container = soup.find("ul", class_="description")

    if container:
        lines = []
        for element in container.descendants:
            tag = getattr(element, "name", None)

            # Section headings (e.g. BACKGROUND, CLAIMS, DETAILED DESCRIPTION)
            if tag == "heading":
                heading_text = element.get_text(strip=True)
                if heading_text:
                    lines.append(f"\n[{heading_text.upper()}]")

            # Actual paragraph content
            elif tag == "div" and "description-line" in element.get("class", []):
                para = element.get_text(separator=" ", strip=True)
                # Strip figure-callout label noise — these are rendered as
                # duplicate text from nested <figure-callout> tags
                para = re.sub(r"\s{2,}", " ", para)
                if para:
                    lines.append(para)

        text = "\n".join(lines)

    else:
        # ------------------------------------------------------------------ #
        # Fallback: no <ul class="description"> found.                        #
        # Strip scripts/styles and take the full body text.                   #
        # This handles edge cases where SerpAPI changes their HTML structure. #
        # ------------------------------------------------------------------ #
        log.warning("    Could not find <ul class='description'> — falling back to body text")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        body = soup.body
        if not body:
            return ""
        text = body.get_text(separator="\n", strip=True)

    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()

    if len(text) > DESCRIPTION_MAX_CHARS:
        text = text[:DESCRIPTION_MAX_CHARS] + "\n\n[TRUNCATED]"

    return text


# ---------------------------------------------------------------------------
# Phase 1 — collect patent IDs for a domain
# ---------------------------------------------------------------------------

def collect_patent_ids(client: serpapi.Client, domain_key: str, query: str) -> list[dict]:
    """
    Single call with num=100 to google_patents engine.
    Filters out scholar results and non-English patents.
    Returns list of raw organic_result dicts.
    """
    log.info("  Querying google_patents — domain='%s'", domain_key)

    data = _sdk_search(client, {
        "engine":   "google_patents",
        "q":        query,
        "num":      10,
        "language": "ENGLISH",
        "type":     "PATENT",      # server-side scholar filter
    })

    if not data:
        log.error("  No data returned for domain '%s'", domain_key)
        return []

    organic = data.get("organic_results", [])
    log.info("  Raw results returned: %d", len(organic))

    filtered = []
    for result in organic:
        # Skip scholar entries — they have no patent_id and no claims
        if result.get("is_scholar") or not result.get("patent_id"):
            continue
        if result.get("language", "en").lower() != "en":
            continue
        filtered.append(result)

    log.info("  After filtering: %d valid patents", len(filtered))
    return filtered


# ---------------------------------------------------------------------------
# Phase 2 — fetch full details for a single patent
# ---------------------------------------------------------------------------

def fetch_details(client: serpapi.Client, patent_id: str) -> Optional[dict]:
    """
    Calls google_patents_details for a single patent_id.
    patent_id is the raw value from organic_results e.g. 'patent/US1234567B2/en'
    """
    return _sdk_search(client, {
        "engine":    "google_patents_details",
        "patent_id": patent_id,
    })


# ---------------------------------------------------------------------------
# Phase 3 — assemble Patent dataclass
# ---------------------------------------------------------------------------

def build_patent(domain: str, meta: dict, details: Optional[dict]) -> Patent:
    pub_number = meta.get("publication_number", "")
    patent_id  = re.sub(r"[^\w]", "_", pub_number)  # file-safe ID

    src = details or {}

    abstract         = src.get("abstract", meta.get("snippet", ""))
    claims           = src.get("claims", [])
    if not isinstance(claims, list):
        claims = []

    description      = ""
    description_link = src.get("description_link", "")
    if description_link:
        log.info("    Scraping description HTML...")
        description = _scrape_description(description_link)

    return Patent(
        patent_id          = patent_id,
        publication_number = pub_number,
        title              = meta.get("title") or src.get("title", ""),
        domain             = domain,
        abstract           = abstract,
        claims             = claims,
        description        = description,
        snippet            = meta.get("snippet", ""),
        inventor           = meta.get("inventor") or src.get("inventor", ""),
        assignee           = meta.get("assignee") or src.get("assignee", ""),
        filing_date        = meta.get("filing_date") or src.get("filing_date", ""),
        grant_date         = meta.get("grant_date") or src.get("grant_date", ""),
        publication_date   = meta.get("publication_date") or src.get("publication_date", ""),
        priority_date      = meta.get("priority_date") or src.get("priority_date", ""),
        patent_link        = meta.get("patent_link", ""),
        language           = meta.get("language", "en"),
        country_status     = meta.get("country_status", {}),
    )


# ---------------------------------------------------------------------------
# Phase 4 — persist to disk
# ---------------------------------------------------------------------------

def save_patent(patent: Patent, output_dir: Path) -> None:
    """
    Writes:
      <output_dir>/json/<patent_id>.json   — structured source of truth
      <output_dir>/txt/<patent_id>.txt     — flat text for embedding pipeline
    """
    (output_dir / "json").mkdir(parents=True, exist_ok=True)
    (output_dir / "txt").mkdir(parents=True, exist_ok=True)

    with open(output_dir / "json" / f"{patent.patent_id}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(patent), f, indent=2, ensure_ascii=False)

    with open(output_dir / "txt" / f"{patent.patent_id}.txt", "w", encoding="utf-8") as f:
        f.write(patent.to_txt())


def save_catalog(patents: list[Patent], output_dir: Path) -> None:
    catalog = [
        {
            "patent_id":          p.patent_id,
            "publication_number": p.publication_number,
            "title":              p.title,
            "domain":             p.domain,
            "assignee":           p.assignee,
            "filing_date":        p.filing_date,
            "has_abstract":       bool(p.abstract),
            "has_claims":         bool(p.claims),
            "has_description":    bool(p.description),
            "claim_count":        len(p.claims),
        }
        for p in patents
    ]
    path = output_dir / "catalog.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    log.info("Catalog saved → %s (%d entries)", path, len(catalog))


def load_cached_ids(output_dir: Path) -> set[str]:
    json_dir = output_dir / "json"
    if not json_dir.exists():
        return set()
    return {p.stem for p in json_dir.glob("*.json")}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(api_key: str, output_dir: Path, dry_run: bool = False) -> None:
    client     = serpapi.Client(api_key=api_key)
    cached_ids = load_cached_ids(output_dir)
    all_patents: list[Patent] = []
    stats = {"saved": 0, "cache_hit": 0, "detail_fail": 0, "no_pub_number": 0}

    log.info("Cache: %d patents already downloaded.", len(cached_ids))

    for domain_key, query in DOMAINS.items():
        log.info("")
        log.info("━" * 60)
        log.info("DOMAIN : %s", domain_key)
        log.info("QUERY  : %s", query)
        log.info("━" * 60)

        metadata_list = collect_patent_ids(client, domain_key, query)

        for idx, meta in enumerate(metadata_list, start=1):
            pub_number = meta.get("publication_number", "")
            if not pub_number:
                stats["no_pub_number"] += 1
                continue

            patent_id = re.sub(r"[^\w]", "_", pub_number)

            if patent_id in cached_ids:
                log.info("  [%d/%d] CACHED  %s", idx, len(metadata_list), patent_id)
                stats["cache_hit"] += 1
                continue

            log.info("  [%d/%d] FETCH   %s — %s",
                     idx, len(metadata_list), patent_id, meta.get("title", "")[:55])

            if dry_run:
                log.info("  [DRY RUN] Skipping detail fetch.")
                continue

            details = fetch_details(client, meta["patent_id"])
            if not details:
                log.warning("  Detail fetch failed for %s — saving metadata only.", patent_id)
                stats["detail_fail"] += 1

            patent = build_patent(domain_key, meta, details)
            save_patent(patent, output_dir)
            all_patents.append(patent)
            cached_ids.add(patent_id)
            stats["saved"] += 1

            log.info("    claims=%d  abstract=%d chars  description=%d chars",
                     len(patent.claims), len(patent.abstract), len(patent.description))

    if all_patents:
        save_catalog(all_patents, output_dir)

    log.info("")
    log.info("━" * 60)
    log.info("DONE")
    log.info("  Saved        : %d", stats["saved"])
    log.info("  Cache hits   : %d", stats["cache_hit"])
    log.info("  Detail fails : %d", stats["detail_fail"])
    log.info("Output → %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patent ingestion pipeline")
    parser.add_argument("--api-key",    default=os.getenv("SERPAPI_KEY"),
                        help="SerpAPI key (or set SERPAPI_KEY env var)")
    parser.add_argument("--output-dir", default="../data/raw",
                        help="Root output directory")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Phase 1 only — no detail fetches or HTML scraping")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Provide --api-key or set the SERPAPI_KEY environment variable.")

    run_pipeline(
        api_key    = args.api_key,
        output_dir = Path(args.output_dir),
        dry_run    = args.dry_run,
    )