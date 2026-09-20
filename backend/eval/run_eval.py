"""Retrieval relevance benchmark.

Reports recall@1/3/5 and MRR broken down by query_type. The breakdown is the
point: aggregate numbers hide that formal-title queries (phrased like the BIS
title) are easy while trade-name queries (how procurement officers actually
write) are the hard case the system exists to solve.

    python3 -m eval.run_eval [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.retrieval import StandardsIndex

GOLD_PATH = Path(__file__).parent / "gold_queries.jsonl"
HELDOUT_PATH = Path(__file__).parent / "heldout_queries.jsonl"
DATA_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "standards.json"
K_VALUES = (1, 3, 5)


def load_gold(path: Path = GOLD_PATH):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(index: StandardsIndex, gold):
    per_type = defaultdict(list)
    rows = []

    for case in gold:
        hits = index.search(case["query"], top_k=max(K_VALUES))
        ranked_ids = [h["standard"]["id"] for h in hits]
        expected = set(case["expected_ids"])

        first_rank = next(
            (i + 1 for i, sid in enumerate(ranked_ids) if sid in expected), None
        )
        result = {
            "query": case["query"],
            "query_type": case["query_type"],
            "expected_ids": case["expected_ids"],
            "top_ids": ranked_ids,
            "first_rank": first_rank,
            "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
            **{f"hit@{k}": bool(first_rank and first_rank <= k) for k in K_VALUES},
        }
        rows.append(result)
        per_type[case["query_type"]].append(result)

    return rows, per_type


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    return {
        "n": n,
        **{f"recall@{k}": sum(r[f"hit@{k}"] for r in rows) / n for k in K_VALUES},
        "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="json_out", help="write full results to this path")
    parser.add_argument("--show-misses", action="store_true", help="print every miss")
    parser.add_argument("--heldout", action="store_true",
                        help="use the held-out set (queries written to avoid every alias "
                             "string, so it measures generalisation rather than lexicon recall)")
    parser.add_argument("--gold", help="path to a custom gold set")
    args = parser.parse_args()

    gold_path = Path(args.gold) if args.gold else (HELDOUT_PATH if args.heldout else GOLD_PATH)

    standards = json.loads(DATA_PATH.read_text(encoding="utf-8"))["standards"]
    index = StandardsIndex(standards)
    index.build()

    gold = load_gold(gold_path)
    rows, per_type = evaluate(index, gold)

    overall = summarize(rows)
    print(f"\nCorpus: {len(standards)} standards | Queries: {overall['n']} "
          f"| Set: {gold_path.name}\n")
    header = f"{'query_type':<14} {'n':>3}  {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}"
    print(header)
    print("-" * len(header))
    for qtype in sorted(per_type):
        s = summarize(per_type[qtype])
        print(f"{qtype:<14} {s['n']:>3}  {s['recall@1']:>6.2f} {s['recall@3']:>6.2f} "
              f"{s['recall@5']:>6.2f} {s['mrr']:>6.3f}")
    print("-" * len(header))
    print(f"{'OVERALL':<14} {overall['n']:>3}  {overall['recall@1']:>6.2f} "
          f"{overall['recall@3']:>6.2f} {overall['recall@5']:>6.2f} {overall['mrr']:>6.3f}")

    misses = [r for r in rows if not r["hit@5"]]
    print(f"\nMissed entirely (not in top 5): {len(misses)}")
    if args.show_misses:
        for r in misses:
            print(f"  [{r['query_type']}] {r['query']}")
            print(f"      expected {r['expected_ids']} | got {r['top_ids'][:3]}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"overall": overall,
                        "per_type": {k: summarize(v) for k, v in per_type.items()},
                        "rows": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
