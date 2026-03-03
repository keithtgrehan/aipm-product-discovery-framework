#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structured_rag.chunking import chunk_text
from structured_rag.index_faiss import FaissIndex


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index over PubMed abstract chunks")
    parser.add_argument("--input", type=Path, default=Path("data/corpus/pubmed_fatigue.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input corpus not found: {args.input}")

    docs = load_jsonl(args.input)
    passages: List[str] = []
    metadata: List[Dict[str, object]] = []

    for doc in docs:
        title = str(doc.get("title", "")).strip()
        abstract = str(doc.get("abstract", "")).strip()
        if not abstract:
            continue

        text = f"{title}. {abstract}" if title else abstract
        chunks = chunk_text(text, max_chars=args.max_chars, overlap=args.overlap)

        for chunk_id, chunk in enumerate(chunks):
            passages.append(chunk)
            metadata.append(
                {
                    "doc_id": doc.get("pmid"),
                    "title": title,
                    "year": doc.get("year"),
                    "journal": doc.get("journal"),
                    "url": doc.get("url"),
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    index = FaissIndex(model_name=args.model)
    index.build(passages=passages, metadata=metadata)
    index.save(args.out_dir)

    print(f"Indexed {len(docs)} documents into {len(passages)} passages")
    print(f"Saved index to {args.out_dir}")


if __name__ == "__main__":
    main()
