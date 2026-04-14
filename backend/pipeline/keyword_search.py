"""
Keyword Retrieval Baselines
===========================
Implements two classic retrieval baselines over processed patent text:

1) Inverted index exact-term lookup
2) Boolean query syntax (AND / OR / NOT with parentheses)

Usage:
    python keyword_search.py --build-index
    python keyword_search.py --query "semiconductor AND lithography"
    python keyword_search.py --query "(5g OR telecom) AND NOT satellite"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORD_RE = re.compile(r"[a-z0-9_]+")
QUERY_TOKEN_RE = re.compile(r'"[^"]+"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[A-Za-z0-9_]+', re.IGNORECASE)
OPERATORS = {"AND", "OR", "NOT"}
PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}


def _tokenize_text(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _load_corpus(processed_dir: Path) -> tuple[dict[str, dict], dict[str, str]]:
    doc_txt_dir = processed_dir / "doc_txt"
    meta_dir = processed_dir / "cleaned_json"

    if not doc_txt_dir.exists() or not meta_dir.exists():
        raise FileNotFoundError(
            "Missing processed corpus directories. Expected doc_txt/ and cleaned_json/."
        )

    documents: dict[str, dict] = {}
    texts: dict[str, str] = {}

    for txt_path in sorted(doc_txt_dir.glob("*.txt")):
        patent_id = txt_path.stem
        meta_path = meta_dir / f"{patent_id}.json"
        if not meta_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        documents[patent_id] = {
            "patent_id": patent_id,
            "title": meta.get("title", ""),
            "domain": meta.get("domain", "unknown"),
        }
        texts[patent_id] = text.lower()

    if not documents:
        raise RuntimeError("No valid documents available to index.")

    return documents, texts


def build_inverted_index(processed_dir: Path) -> dict:
    documents, texts = _load_corpus(processed_dir)
    postings: dict[str, set[str]] = {}

    for patent_id, text in texts.items():
        terms = set(_tokenize_text(text))
        for term in terms:
            postings.setdefault(term, set()).add(patent_id)

    serialized_postings = {
        term: sorted(doc_ids) for term, doc_ids in sorted(postings.items())
    }

    index = {
        "documents": documents,
        "texts": texts,
        "postings": serialized_postings,
    }
    return index


def save_index(index: dict, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    log.info("Saved inverted index to %s", index_path)


def load_index(index_path: Path) -> dict:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_tokens(query: str) -> list[str]:
    tokens = QUERY_TOKEN_RE.findall(query)
    if not tokens:
        raise ValueError("Empty query")
    return tokens


def _to_rpn(tokens: list[str]) -> list[str]:
    output: list[str] = []
    stack: list[str] = []

    for token in tokens:
        upper = token.upper()

        if token == "(":
            stack.append(token)
        elif token == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("Mismatched parentheses in query")
            stack.pop()
        elif upper in OPERATORS:
            while stack and stack[-1] != "(":
                top = stack[-1]
                if PRECEDENCE.get(top, 0) >= PRECEDENCE[upper]:
                    output.append(stack.pop())
                else:
                    break
            stack.append(upper)
        else:
            output.append(token)

    while stack:
        top = stack.pop()
        if top in {"(", ")"}:
            raise ValueError("Mismatched parentheses in query")
        output.append(top)

    return output


def _phrase_match_docs(phrase: str, index: dict) -> set[str]:
    terms = _tokenize_text(phrase)
    if not terms:
        return set()

    postings = index["postings"]
    candidate = set(postings.get(terms[0], []))
    for term in terms[1:]:
        candidate &= set(postings.get(term, []))
        if not candidate:
            return set()

    texts = index["texts"]
    phrase_lc = phrase.lower()
    return {doc_id for doc_id in candidate if phrase_lc in texts.get(doc_id, "")}


def _lookup_token(token: str, index: dict) -> set[str]:
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        return _phrase_match_docs(token[1:-1], index)

    term = token.lower()
    return set(index["postings"].get(term, []))


def execute_boolean_query(query: str, index: dict) -> list[dict]:
    universe = set(index["documents"].keys())
    rpn = _to_rpn(_query_tokens(query))
    stack: list[set[str]] = []

    for token in rpn:
        if token == "NOT":
            if not stack:
                raise ValueError("Invalid query: NOT missing operand")
            stack.append(universe - stack.pop())
        elif token in {"AND", "OR"}:
            if len(stack) < 2:
                raise ValueError(f"Invalid query: {token} missing operand")
            right = stack.pop()
            left = stack.pop()
            stack.append(left & right if token == "AND" else left | right)
        else:
            stack.append(_lookup_token(token, index))

    if len(stack) != 1:
        raise ValueError("Invalid query syntax")

    matched_ids = sorted(stack.pop())
    return [index["documents"][doc_id] for doc_id in matched_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverted-index and Boolean query baselines")
    parser.add_argument(
        "--processed-dir",
        default=str(REPO_ROOT / "data" / "processed"),
        help="Processed corpus root containing doc_txt and cleaned_json",
    )
    parser.add_argument(
        "--index-path",
        default=str(REPO_ROOT / "data" / "processed" / "inverted_index.json"),
        help="Path to saved inverted index JSON",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build and persist inverted index",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Boolean query, e.g. '(5g OR telecom) AND NOT satellite'",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Max results to print")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    index_path = Path(args.index_path).resolve()

    index: dict
    if args.build_index or not index_path.exists():
        log.info("Building inverted index from %s", processed_dir)
        index = build_inverted_index(processed_dir)
        save_index(index, index_path)
    else:
        index = load_index(index_path)

    if args.query:
        results = execute_boolean_query(args.query, index)
        payload = {
            "query": args.query,
            "num_results": len(results),
            "results": results[: args.top_k],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
