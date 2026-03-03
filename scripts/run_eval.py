#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from structured_rag.index_faiss import FaissIndex
from structured_rag.llm import generate_answer
from structured_rag.prompts import baseline_prompt, structured_prompt
from structured_rag.retrieve import retrieve_top_k
from structured_rag.verify import score_omission


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_context(passages: List[Dict[str, object]]) -> str:
    blocks: List[str] = []
    for i, p in enumerate(passages, start=1):
        snippet = str(p.get("text", "")).replace("\n", " ").strip()
        title = str(p.get("title", ""))
        year = p.get("year")
        pmid = p.get("doc_id")
        blocks.append(f"[p{i}] {snippet}\nSource: {title} ({year}), PMID {pmid}")
    return "\n\n".join(blocks)


def rel_delta_pct(baseline: float, structured: float) -> float:
    if baseline == 0:
        return 0.0 if structured == 0 else -100.0
    return ((baseline - structured) / baseline) * 100.0


def bool_label(flag: bool) -> str:
    return "true" if flag else "false"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run omission benchmark for baseline vs structured prompts")
    parser.add_argument("--llm", choices=["ollama", "groq"], default="ollama")
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of benchmark queries")
    parser.add_argument("--timeout", type=int, default=180, help="LLM request timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=400, help="Max generated tokens per answer")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--queries", type=Path, default=Path("benchmarks/queries.jsonl"))
    parser.add_argument("--checklists", type=Path, default=Path("benchmarks/checklists.jsonl"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/results.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("reports/summary.md"))
    args = parser.parse_args()

    queries = load_jsonl(args.queries)
    if args.limit is not None:
        queries = queries[: max(0, args.limit)]
    checklists = {row["id"]: row for row in load_jsonl(args.checklists)}
    index = FaissIndex.load(args.index_dir)

    results: List[Dict[str, object]] = []

    for row in queries:
        qid = str(row["id"])
        question = str(row["query"])
        checklist = checklists[qid]

        passages = retrieve_top_k(index=index, question=question, top_k=args.top_k)
        context = to_context(passages)

        base_prompt = baseline_prompt(context=context, question=question)
        start = perf_counter()
        base_answer = generate_answer(
            prompt=base_prompt,
            llm=args.llm,
            model=args.model,
            temperature=0.0,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        latency_base_ms = (perf_counter() - start) * 1000.0

        struct_prompt = structured_prompt(context=context, question=question, checklist=checklist)
        start = perf_counter()
        struct_answer = generate_answer(
            prompt=struct_prompt,
            llm=args.llm,
            model=args.model,
            temperature=0.0,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        latency_struct_ms = (perf_counter() - start) * 1000.0

        base_score = score_omission(base_answer, checklist)
        struct_score = score_omission(struct_answer, checklist)

        omission_base = float(base_score["omission_score"])
        omission_struct = float(struct_score["omission_score"])

        results.append(
            {
                "id": qid,
                "omission_baseline": omission_base,
                "omission_structured": omission_struct,
                "delta_rel_pct": rel_delta_pct(omission_base, omission_struct),
                "fail_baseline": bool(base_score["binary_fail"]),
                "fail_structured": bool(struct_score["binary_fail"]),
                "latency_baseline_ms": latency_base_ms,
                "latency_structured_ms": latency_struct_ms,
            }
        )

        print(
            f"{qid}: omission baseline={omission_base:.3f}, structured={omission_struct:.3f}, "
            f"latency_ms baseline={latency_base_ms:.1f}, structured={latency_struct_ms:.1f}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "omission_baseline",
                "omission_structured",
                "delta_rel_pct",
                "fail_baseline",
                "fail_structured",
                "latency_baseline_ms",
                "latency_structured_ms",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "id": r["id"],
                    "omission_baseline": f"{float(r['omission_baseline']):.6f}",
                    "omission_structured": f"{float(r['omission_structured']):.6f}",
                    "delta_rel_pct": f"{float(r['delta_rel_pct']):.2f}",
                    "fail_baseline": bool_label(bool(r["fail_baseline"])),
                    "fail_structured": bool_label(bool(r["fail_structured"])),
                    "latency_baseline_ms": f"{float(r['latency_baseline_ms']):.2f}",
                    "latency_structured_ms": f"{float(r['latency_structured_ms']):.2f}",
                }
            )

    mean_base = statistics.fmean(float(r["omission_baseline"]) for r in results)
    mean_struct = statistics.fmean(float(r["omission_structured"]) for r in results)
    rel_improvement = rel_delta_pct(mean_base, mean_struct)

    median_lat_base = statistics.median(float(r["latency_baseline_ms"]) for r in results)
    median_lat_struct = statistics.median(float(r["latency_structured_ms"]) for r in results)
    latency_overhead = (
        ((median_lat_struct - median_lat_base) / median_lat_base) * 100.0
        if median_lat_base > 0
        else 0.0
    )

    omission_target_pass = rel_improvement >= 20.0
    latency_target_pass = latency_overhead <= 10.0

    summary = f"""# Experiment 1 Summary

- Mean omission (baseline): {mean_base:.4f}
- Mean omission (structured): {mean_struct:.4f}
- Relative omission improvement: {rel_improvement:.2f}%
- Median latency baseline: {median_lat_base:.2f} ms
- Median latency structured: {median_lat_struct:.2f} ms
- Latency overhead: {latency_overhead:.2f}%

## Targets

- Omission improvement >= 20%: {'PASS' if omission_target_pass else 'FAIL'}
- Latency overhead <= 10%: {'PASS' if latency_target_pass else 'FAIL'}
"""

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(summary, encoding="utf-8")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.summary_md}")


if __name__ == "__main__":
    main()
