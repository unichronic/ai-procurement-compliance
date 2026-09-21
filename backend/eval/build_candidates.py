"""Dump retrieval candidates for relabelling the gold sets.

The original labels were written against a 39-standard corpus, where each
query had exactly one plausible answer. At 6,383 standards that assumption is
false: "rolled girders and joists" legitimately matches both IS 2062
(structural steel) and IS 808 (beam and joist dimensions), and a single-label
gold set scores the second as a miss.

This dumps each query's top-N candidates so they can be judged, producing a
multi-label set. Judging a fixed retrieved pool is itself biased — a correct
standard the engine never retrieves can't be labelled — so recall figures from
the relabelled set are an upper bound, and that is recorded in the output.

    python3 -m eval.build_candidates --out eval/candidates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.rerank import CrossEncoderReranker
from app.core.retrieval import StandardsIndex
from eval.run_eval import DATA_PATH, GOLD_PATH, HELDOUT_PATH, load_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/candidates.json")
    ap.add_argument("--depth", type=int, default=20)
    args = ap.parse_args()

    standards = json.loads(DATA_PATH.read_text(encoding="utf-8"))["standards"]
    index = StandardsIndex(standards)
    index.build()
    reranker = CrossEncoderReranker()

    out = []
    for path, set_name in ((GOLD_PATH, "gold"), (HELDOUT_PATH, "heldout")):
        for case in load_gold(path):
            hits = index.search(case["query"], top_k=args.depth)
            if reranker.available:
                hits = reranker.rerank(case["query"], hits, top_k=args.depth)

            out.append({
                "set": set_name,
                "query": case["query"],
                "query_type": case["query_type"],
                "original_expected_ids": case["expected_ids"],
                "candidates": [
                    {
                        "id": h["standard"]["id"],
                        "number": h["standard"]["number"],
                        "title": h["standard"]["title"],
                        "scope": (h["standard"].get("scope") or "")[:220],
                    }
                    for h in hits
                ],
            })

    Path(args.out).write_text(
        json.dumps({
            "corpus_size": len(standards),
            "depth": args.depth,
            "note": ("Candidates are what the current engine retrieves. A correct "
                     "standard never retrieved cannot be judged here, so recall "
                     "measured against the relabelled set is an upper bound."),
            "queries": out,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}: {len(out)} queries x {args.depth} candidates")


if __name__ == "__main__":
    main()
