"""Build the allied-standards graph from the cached archive documents.

The linter's `missing_allied_standard` rule tells an officer that a standard
they cited normatively depends on another they omitted. It reasons over
hand-authored edges covering 17 records out of 6,383, which makes it
decorative rather than useful.

`data/archive/cache/` holds 9,411 raw standard texts, each with the annex
listing its referred Indian Standards. That is the same relationship, stated
by the documents themselves, so the graph can be read rather than written.

Edges are only created between standards that are both in the corpus; a
reference to something not indexed is counted and reported rather than
silently dropped, because "this standard cites 14 others and we hold 9 of
them" is a coverage fact worth knowing.

    python3 -m data_pipeline.build_allied_graph --write
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_pipeline.extract import extract_clause2_references, extract_referred_standards
from data_pipeline.mirror import parse_item_id

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "archive" / "cache"
STANDARDS = ROOT / "backend" / "app" / "data" / "standards.json"

_NUMBER_RE = re.compile(
    r"IS(?:/IEC|/ISO)?\s*(?P<base>\d{2,5})"
    r"(?:\s*\(\s*Part\s*(?P<part>\d+)[^)]*\))?",
    re.IGNORECASE,
)


def corpus_lookup(standards: List[Dict[str, Any]]):
    """(base, part) -> the active/newest record, so references resolve to a
    citable edition rather than an arbitrary one."""
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in standards:
        m = _NUMBER_RE.match(s.get("number", ""))
        if not m:
            continue
        key = (m.group("base"), m.group("part") or "")
        current = by_key.get(key)
        if current is None:
            by_key[key] = s
            continue
        # Prefer active, then the later edition.
        better = (
            (s.get("status") == "active", s.get("edition_year") or 0)
            > (current.get("status") == "active", current.get("edition_year") or 0)
        )
        if better:
            by_key[key] = s
    return by_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap files processed")
    args = ap.parse_args()

    doc = json.loads(STANDARDS.read_text(encoding="utf-8"))
    standards = doc["standards"]
    by_key = corpus_lookup(standards)
    by_id = {s["id"]: s for s in standards}

    files = sorted(CACHE_DIR.glob("*.txt"))
    if args.limit:
        files = files[: args.limit]

    stats = Counter()
    edges_by_source: Dict[str, List[Dict[str, str]]] = {}

    for path in files:
        stats["files"] += 1
        item = parse_item_id(path.stem)
        if not item:
            stats["unparseable_filename"] += 1
            continue

        source = by_key.get((item["number"], item.get("part") or ""))
        if source is None:
            stats["source_not_in_corpus"] += 1
            continue

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stats["unreadable"] += 1
            continue

        # Two legitimate sources: the referred-standards annex, and clause 2
        # where a document lists its references inline instead. Only about one
        # document in forty has the annex.
        refs = extract_referred_standards(raw, self_number=item["number"])
        if refs:
            stats["source_annex"] += 1
        clause2 = extract_clause2_references(raw, self_number=item["number"])
        if clause2 and not refs:
            stats["source_clause2"] += 1
        merged = {(r["number"], r.get("part") or ""): r for r in clause2}
        merged.update({(r["number"], r.get("part") or ""): r for r in refs})
        refs = list(merged.values())

        if not refs:
            continue
        stats["files_with_refs"] += 1

        edges = []
        seen = set()
        for ref in refs:
            stats["references_found"] += 1
            target = by_key.get((ref["number"], ref.get("part") or ""))
            if target is None:
                stats["reference_outside_corpus"] += 1
                continue
            if target["id"] == source["id"] or target["id"] in seen:
                continue
            seen.add(target["id"])
            edges.append({
                "id": target["id"],
                # The annex states a normative dependency; it does not classify
                # it further, so claiming a finer type would be invention.
                "type": "normative_reference",
                "note": f"Listed in the referred-standards annex of {source['number']}.",
                "source": "extracted_from_annex",
            })

        if edges:
            edges_by_source.setdefault(source["id"], []).extend(edges)
            stats["edges_created"] += len(edges)

    print(f"cached files scanned      : {stats['files']}")
    print(f"  source not in corpus    : {stats['source_not_in_corpus']}")
    print(f"  files with an annex list: {stats['files_with_refs']}")
    print(f"references found          : {stats['references_found']}")
    print(f"  resolved to corpus      : {stats['edges_created']}")
    print(f"  outside corpus (kept as coverage fact): {stats['reference_outside_corpus']}")
    print(f"standards gaining edges   : {len(edges_by_source)} "
          f"({100 * len(edges_by_source) / len(standards):.1f}% of corpus)")

    if args.write:
        for sid, edges in edges_by_source.items():
            record = by_id[sid]
            existing = record.get("allied") or []
            # Curated edges win; extracted ones fill the gaps.
            curated_targets = {e["id"] for e in existing if e.get("source") != "extracted_from_annex"}
            merged = [e for e in existing if e.get("source") != "extracted_from_annex"]
            merged += [e for e in edges if e["id"] not in curated_targets]
            record["allied"] = merged

        doc["standards"] = standards
        doc.setdefault("_meta", {})["allied_graph"] = {
            "edges_extracted": stats["edges_created"],
            "standards_with_edges": len(edges_by_source),
            "method": "referred-standards annex of the archive full text",
            "caveat": ("Annex lists are read from scanned text; a reference the "
                       "parser could not attribute confidently is dropped rather "
                       "than guessed."),
        }
        STANDARDS.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        print(f"\nWrote {STANDARDS}")
    else:
        print("\n(dry run — pass --write)")


if __name__ == "__main__":
    main()
