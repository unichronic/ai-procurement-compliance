"""Merge the 6,360-standard archive corpus with the curated records.

The team's `data/standards_corpus_full.json` is the first real answer to this
project's binding constraint: 6,264 of its records carry scope text extracted
from published documents rather than written by hand. What it has none of is
the curated layer this engine's rules depend on — QCO/certification mappings,
trade-name aliases, technical designations, and allied-standard edges.

So neither corpus supersedes the other. This merges them: the archive supplies
breadth, the curated records supply the fields the linter reasons over, and
where both describe the same standard the curated fields win.

    python3 -m data_pipeline.merge_full_corpus --write
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
FULL_CORPUS = ROOT / "data" / "standards_corpus_full.json"
CURATED = ROOT / "backend" / "app" / "data" / "standards.json"

# Their sector codes -> the department names this engine already reports.
CATEGORY_TO_DEPARTMENT = {
    "cement_building_materials": "Civil Engineering",
    "chemicals": "Chemical Engineering",
    "electrical_cables": "Electrotechnical",
    "electrical_installations": "Electrotechnical",
    "food_agriculture": "Food and Agriculture",
    "geotechnical": "Civil Engineering",
    "machinery_equipment": "Production and General Engineering",
    "measurement_testing": "Production and General Engineering",
    "packaging": "Production and General Engineering",
    "plastic_pipes": "Civil Engineering",
    "ppe": "Production and General Engineering",
    "rubber_leather": "Chemical Engineering",
    "steel_pipes_fittings": "Metallurgical Engineering",
    "structural_steel": "Metallurgical Engineering",
    "textiles": "Textile Engineering",
    "timber_furniture": "Civil Engineering",
    "water_quality": "Chemical Engineering",
}

_NUMBER_RE = re.compile(
    r"IS\s*(?:/\s*(?:IEC|ISO))?\s*(?P<base>\d{2,5})"
    r"(?:\s*\(\s*Part\s*(?P<part>\d+)[^)]*\))?"
    r"(?:\s*:\s*(?P<year>(?:18|19|20)\d{2}))?",
    re.IGNORECASE,
)

# Their extractor leaves the edition year on the front of some titles
# ("2013: Geotechnical Investigations ..."), which would then be embedded and
# shown to users as part of the name.
_TITLE_YEAR_PREFIX_RE = re.compile(r"^\s*(?:18|19|20)\d{2}\s*[:\-]\s*")


def parse_number(number: str) -> Optional[Dict[str, Any]]:
    m = _NUMBER_RE.search(number or "")
    if not m:
        return None
    return {
        "base": m.group("base"),
        "part": m.group("part"),
        "year": int(m.group("year")) if m.group("year") else None,
    }


def match_key(base: str, part: Optional[str], year: Optional[int]) -> Tuple:
    return (str(base), str(part) if part else None, year)


def clean_title(title: str) -> str:
    return _TITLE_YEAR_PREFIX_RE.sub("", title or "").strip()


def make_id(base: str, part: Optional[str], year: Optional[int]) -> str:
    bits = ["IS", str(base)]
    if part:
        bits.append(str(part))
    if year:
        bits.append(str(year))
    return "_".join(bits)


def to_engine_schema(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    parsed = parse_number(record.get("number", ""))
    if not parsed:
        return None

    base, part, year = parsed["base"], parsed["part"], parsed["year"]
    display = f"IS {base}" + (f" (Part {part})" if part else "")

    return {
        "id": make_id(base, part, year),
        "number": display,
        "edition_year": year,
        "title": clean_title(record.get("title", "")),
        "scope": (record.get("scope") or "").strip(),
        "committee": record.get("committee") or "",
        "department": CATEGORY_TO_DEPARTMENT.get(record.get("category"), "Unclassified"),
        "status": record.get("status") or "active",
        "superseded_by": None,          # resolved below, once ids exist
        "supersedes": None,
        "amendments": [],
        "allied": [],
        "qco": None,
        "aliases": [],
        "designations": [],
        # Their own README is explicit that none of this is checked against the
        # BIS catalogue, so it must not claim to be.
        "data_confidence": "medium" if record.get("provenance") == "published_text_ocr" else "low",
        "verified": bool(record.get("verified")),
        "provenance": record.get("provenance"),
        "source_url": record.get("source_url"),
    }


CURATED_FIELDS = ("aliases", "designations", "qco", "allied", "amendments",
                  "superseded_by", "supersedes", "committee", "chunks",
                  "referred_standards", "source")


def merge(full: List[Dict[str, Any]], curated: List[Dict[str, Any]]):
    converted = []
    for record in full:
        engine = to_engine_schema(record)
        if engine:
            converted.append((engine, record))

    by_key: Dict[Tuple, Dict[str, Any]] = {}
    order: List[Tuple] = []
    for engine, _ in converted:
        parsed = parse_number(engine["number"])
        key = match_key(parsed["base"], parsed["part"], engine["edition_year"])
        if key in by_key:
            continue  # first wins; duplicates are the same edition twice
        by_key[key] = engine
        order.append(key)

    stats = Counter()
    for record in curated:
        parsed = parse_number(record["number"])
        if not parsed:
            stats["curated_unparseable"] += 1
            continue
        key = match_key(parsed["base"], parsed["part"], record.get("edition_year"))

        target = by_key.get(key)
        if target is None:
            # Curated standard the archive doesn't carry -- keep it wholesale.
            by_key[key] = dict(record)
            order.append(key)
            stats["curated_only"] += 1
            continue

        # Overlay: the archive's extracted scope is kept, curation wins on the
        # fields the rules depend on, and the curated id is preserved so the
        # existing benchmark and citation tests keep resolving.
        merged = dict(target)
        merged["id"] = record["id"]
        merged["title"] = record.get("title") or merged["title"]
        for field in CURATED_FIELDS:
            if record.get(field):
                merged[field] = record[field]
        if record.get("scope") and not merged.get("scope"):
            merged["scope"] = record["scope"]
        merged["data_confidence"] = record.get("data_confidence", merged["data_confidence"])
        by_key[key] = merged
        stats["overlaid"] += 1

    merged_list = [by_key[k] for k in order]

    # Resolve supersession now that every record has an id.
    id_by_key = {}
    for rec in merged_list:
        parsed = parse_number(rec["number"])
        if parsed:
            id_by_key[match_key(parsed["base"], parsed["part"], rec.get("edition_year"))] = rec["id"]

    for engine, raw in converted:
        target = by_key.get(match_key(*_key_of(engine)))
        if target is None or target.get("superseded_by"):
            continue
        succ = raw.get("superseded_by_number")
        if not succ:
            continue
        sp = parse_number(succ)
        if not sp:
            continue
        succ_id = id_by_key.get(match_key(sp["base"], sp["part"], sp["year"]))
        if succ_id:
            target["superseded_by"] = [succ_id]
            stats["supersession_resolved"] += 1

    return merged_list, stats


def _key_of(engine: Dict[str, Any]) -> Tuple:
    parsed = parse_number(engine["number"])
    return (parsed["base"], parsed["part"], engine["edition_year"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=str(CURATED))
    args = ap.parse_args()

    raw = json.loads(FULL_CORPUS.read_text(encoding="utf-8"))
    full = raw if isinstance(raw, list) else (raw.get("standards") or list(raw.values())[0])
    curated_doc = json.loads(CURATED.read_text(encoding="utf-8"))
    curated = curated_doc["standards"]

    merged, stats = merge(full, curated)

    print(f"archive records      : {len(full)}")
    print(f"curated records      : {len(curated)}")
    print(f"  overlaid onto archive: {stats['overlaid']}")
    print(f"  kept (not in archive): {stats['curated_only']}")
    print(f"  unparseable numbers  : {stats['curated_unparseable']}")
    print(f"supersession resolved: {stats['supersession_resolved']}")
    print(f"merged total         : {len(merged)}")
    print(f"  with aliases       : {sum(1 for r in merged if r.get('aliases'))}")
    print(f"  with qco           : {sum(1 for r in merged if r.get('qco'))}")
    print(f"  with allied edges  : {sum(1 for r in merged if r.get('allied'))}")
    print(f"  with scope text    : {sum(1 for r in merged if (r.get('scope') or '').strip())}")

    if args.write:
        curated_doc["standards"] = merged
        curated_doc.setdefault("_meta", {})["corpus"] = {
            "archive_records": len(full),
            "curated_overlaid": stats["overlaid"],
            "total": len(merged),
            "note": ("Archive records carry scope extracted from published text but are "
                     "unverified against the BIS catalogue. Curated records supply "
                     "certification, alias and allied-standard data."),
        }
        Path(args.out).write_text(
            json.dumps(curated_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    else:
        print("\n(dry run — pass --write)")


if __name__ == "__main__":
    main()
