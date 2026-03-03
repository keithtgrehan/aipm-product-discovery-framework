#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_OUTPUT = Path("data/corpus/pubmed_fatigue.jsonl")

SEARCH_TERMS = [
    "fatigue differential diagnosis",
    "chronic fatigue red flags",
    "fatigue anemia hypothyroidism",
    "fatigue sleep apnea depression",
]


def _text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return " ".join("".join(elem.itertext()).split())


def _extract_year(article: ET.Element) -> int | None:
    candidates = [
        article.findtext(".//Article/ArticleDate/Year"),
        article.findtext(".//Article/Journal/JournalIssue/PubDate/Year"),
        article.findtext(".//DateCompleted/Year"),
    ]
    for val in candidates:
        if val and val.isdigit():
            return int(val)

    medline = article.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate") or ""
    match = re.search(r"(19|20)\d{2}", medline)
    if match:
        return int(match.group(0))
    return None


def esearch(term: str, retmax: int, email: str | None = None) -> List[str]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    if email:
        params["email"] = email

    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("esearchresult", {}).get("idlist", [])


def efetch(pmids: List[str], email: str | None = None) -> List[Dict[str, object]]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email

    resp = requests.get(EFETCH_URL, params=params, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    rows: List[Dict[str, object]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID")
        if not pmid:
            continue

        title = _text(article.find(".//Article/ArticleTitle"))
        journal = _text(article.find(".//Article/Journal/Title"))

        abstract_parts: List[str] = []
        for abs_node in article.findall(".//Article/Abstract/AbstractText"):
            segment = _text(abs_node)
            if not segment:
                continue
            label = abs_node.attrib.get("Label") or abs_node.attrib.get("NlmCategory")
            if label:
                abstract_parts.append(f"{label}: {segment}")
            else:
                abstract_parts.append(segment)

        abstract = " ".join(abstract_parts).strip()
        if not abstract:
            continue

        year = _extract_year(article)

        rows.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )

    return rows


def batched(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch fatigue-domain PubMed abstracts into JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int, default=400, help="Target abstract count")
    parser.add_argument("--per-query", type=int, default=150, help="esearch retmax per search term")
    parser.add_argument("--batch-size", type=int, default=100, help="efetch pmid batch size")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument("--email", type=str, default=None, help="Optional contact email for NCBI")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Exists, skipping: {args.output}")
        return

    seen_pmids = set()
    ordered_pmids: List[str] = []

    for term in SEARCH_TERMS:
        ids = esearch(term, retmax=args.per_query, email=args.email)
        for pmid in ids:
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            ordered_pmids.append(pmid)
            if len(ordered_pmids) >= args.max_records:
                break
        if len(ordered_pmids) >= args.max_records:
            break
        time.sleep(0.34)

    target_pmids = ordered_pmids[: args.max_records]
    print(f"Collected {len(target_pmids)} candidate PMIDs")

    rows: List[Dict[str, object]] = []
    seen_rows = set()

    for chunk in batched(target_pmids, args.batch_size):
        fetched = efetch(chunk, email=args.email)
        for row in fetched:
            pmid = row["pmid"]
            if pmid in seen_rows:
                continue
            seen_rows.add(pmid)
            rows.append(row)
        time.sleep(0.34)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} abstracts -> {args.output}")


if __name__ == "__main__":
    main()
