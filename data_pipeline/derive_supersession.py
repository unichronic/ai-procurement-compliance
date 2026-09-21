"""Derive supersession from edition years.

The archive import carries almost no supersession data: 656 IS numbers appear
with two or more editions and every one of them is marked `active`. IS 10086
exists as both 1982 and 2021, both "current". That makes the linter's flagship
rule — telling an officer they have cited a withdrawn edition — structurally
unable to fire on the overwhelming majority of the corpus.

BIS practice is that a new edition of a number supersedes the previous one, so
within a family (same number, same part) every edition but the newest is
superseded by it. That is a mechanical inference from data already present, not
new research.

What it deliberately does NOT do:

* It never contradicts curated data. A record whose status or superseded_by was
  set by hand or by a real document keeps it.
* It never infers across numbers. IS 226 -> IS 2062 is a real supersession that
  only a document states; nothing here invents that kind of edge.
* It marks the inference on each record (`supersession_source: derived`) so the
  API can say the difference out loud, because "superseded, per BIS catalogue"
  and "superseded, because we saw a newer edition" are different claims.

    python3 -m data_pipeline.derive_supersession --write
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
STANDARDS = ROOT / "backend" / "app" / "data" / "standards.json"

_FAMILY_RE = re.compile(
    r"^(?P<prefix>IS(?:/IEC|/ISO)?)\s*(?P<base>\d{2,5})"
    r"(?:\s*\(\s*Part\s*(?P<part>\d+)(?:\s*/\s*Sec(?:tion)?\.?\s*(?P<sec>\d+))?[^)]*\))?",
    re.IGNORECASE,
)


def family_key(number: str) -> Tuple[str, str, str] | None:
    m = _FAMILY_RE.match((number or "").strip())
    if not m:
        return None
    return (m.group("base"), m.group("part") or "", m.group("sec") or "")


def derive(standards: List[Dict[str, Any]]) -> Dict[str, Any]:
    families: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for record in standards:
        key = family_key(record.get("number", ""))
        if key and record.get("edition_year"):
            families[key].append(record)

    stats = {
        "families_examined": 0,
        "records_marked": 0,
        "left_alone_curated": 0,
        "families_ambiguous": 0,
    }

    for key, members in families.items():
        if len(members) < 2:
            continue
        stats["families_examined"] += 1

        newest = max(members, key=lambda r: r["edition_year"])
        # Two records claiming the same number AND year are a duplicate import,
        # not a supersession; leave them for a human.
        if sum(1 for m in members if m["edition_year"] == newest["edition_year"]) > 1:
            stats["families_ambiguous"] += 1
            continue

        for record in members:
            if record is newest:
                continue
            # Curated or document-sourced supersession wins over inference.
            if record.get("superseded_by") or record.get("supersession_source") == "curated":
                stats["left_alone_curated"] += 1
                continue
            if record.get("status") not in (None, "", "active"):
                stats["left_alone_curated"] += 1
                continue

            record["status"] = "superseded"
            record["superseded_by"] = [newest["id"]]
            record["supersession_source"] = "derived_from_edition_year"
            stats["records_marked"] += 1

        supersedes = [m["id"] for m in members if m is not newest]
        if supersedes and not newest.get("supersedes"):
            newest["supersedes"] = supersedes

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    doc = json.loads(STANDARDS.read_text(encoding="utf-8"))
    standards = doc["standards"]

    before_superseded = sum(1 for s in standards if s.get("status") != "active")
    stats = derive(standards)
    after_superseded = sum(1 for s in standards if s.get("status") != "active")

    print(f"families with multiple editions : {stats['families_examined']}")
    print(f"  records marked superseded     : {stats['records_marked']}")
    print(f"  left alone (curated/explicit) : {stats['left_alone_curated']}")
    print(f"  ambiguous (duplicate year)    : {stats['families_ambiguous']}")
    print(f"non-active records: {before_superseded} -> {after_superseded} "
          f"({100 * after_superseded / len(standards):.1f}% of corpus)")

    if args.write:
        doc["standards"] = standards
        doc.setdefault("_meta", {})["supersession"] = {
            "derived_records": stats["records_marked"],
            "method": "newest edition within a number/part supersedes older ones",
            "caveat": ("Inferred from edition years present in the corpus, not read "
                       "from the BIS catalogue. Cross-number supersession "
                       "(IS 226 -> IS 2062) is not inferred."),
        }
        STANDARDS.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        print(f"\nWrote {STANDARDS}")
    else:
        print("\n(dry run — pass --write)")


if __name__ == "__main__":
    main()
