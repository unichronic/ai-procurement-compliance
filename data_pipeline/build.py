"""Ingestion CLI.

    # refresh the standards already in the corpus from their real documents
    python3 -m data_pipeline.build --refresh-existing

    # fetch specific standards by archive.org item id
    python3 -m data_pipeline.build --items gov.in.is.1786.2008 gov.in.is.1239.1.2004

    # walk the mirror's sectional indexes and report catalogue size
    python3 -m data_pipeline.build --catalogue

Nothing is written unless --write is passed, so a run can always be inspected
first.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_pipeline import corpus as corpus_mod
from data_pipeline import mirror
from data_pipeline.extract import extract_standard

ROOT = Path(__file__).resolve().parents[1]
STANDARDS_PATH = ROOT / "backend" / "app" / "data" / "standards.json"
SNAPSHOT_PATH = ROOT / "data_pipeline" / "snapshot.json"
CACHE_DIR = ROOT / "data_pipeline" / "cache"


def _load_standards() -> Dict[str, Any]:
    return json.loads(STANDARDS_PATH.read_text(encoding="utf-8"))


def _item_id_for(record: Dict[str, Any]) -> Optional[str]:
    """Reconstruct the mirror item id for an existing corpus record."""
    number = record.get("number", "")
    digits = "".join(ch for ch in number.split("(")[0] if ch.isdigit())
    if not digits or not record.get("edition_year"):
        return None

    bits = ["gov.in.is", digits]
    part = None
    if "(" in number:
        inner = number.split("(", 1)[1]
        part_digits = "".join(ch for ch in inner.split("/")[0] if ch.isdigit())
        part = part_digits or None
    if part:
        bits.append(part)
    bits.append(str(record["edition_year"]))
    return ".".join(bits)


def _fetch(item_id: str, use_cache: bool = True) -> Optional[str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{item_id}.txt"
    if use_cache and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")

    raw = mirror.fetch_standard_text(item_id)
    if raw:
        cached.write_text(raw, encoding="utf-8")
    return raw


def ingest_items(item_ids: List[str], *, delay: float = 1.0,
                 use_cache: bool = True) -> List[Dict[str, Any]]:
    out = []
    for item_id in item_ids:
        parsed = mirror.parse_item_id(item_id)
        if not parsed:
            print(f"  skip  {item_id}: unparseable item id")
            continue

        raw = _fetch(item_id, use_cache=use_cache)
        if not raw:
            print(f"  miss  {item_id}: not available on mirror")
            continue

        record = extract_standard(raw, item=parsed)
        status = "ok" if record.get("scope") else "thin"
        print(f"  {status:5} {record['id']:22} {(record.get('title') or '')[:52]}")
        out.append(record)
        time.sleep(delay)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-existing", action="store_true",
                    help="re-fetch every standard already in the corpus")
    ap.add_argument("--items", nargs="*", help="archive.org item ids to ingest")
    ap.add_argument("--catalogue", action="store_true",
                    help="walk sectional indexes and report catalogue size")
    ap.add_argument("--sections", nargs="*", default=None)
    ap.add_argument("--write", action="store_true", help="persist changes")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.catalogue:
        sections = args.sections or mirror.SECTIONS
        print(f"Walking {len(sections)} sectional indexes...")
        entries = mirror.fetch_catalogue(sections, delay=args.delay)
        print(f"\nCatalogue entries discovered: {len(entries)}")
        by_division: Dict[str, int] = {}
        for e in entries:
            by_division[e["division"]] = by_division.get(e["division"], 0) + 1
        for div in sorted(by_division):
            print(f"  {div}: {by_division[div]}")
        if args.write:
            out = ROOT / "data_pipeline" / "catalogue.json"
            out.write_text(json.dumps(
                {"fetched_at": datetime.now(timezone.utc).isoformat(),
                 "count": len(entries), "entries": entries},
                indent=2), encoding="utf-8")
            print(f"\nWrote {out}")
        return

    data = _load_standards()
    existing = data["standards"]

    item_ids: List[str] = list(args.items or [])
    if args.refresh_existing:
        for record in existing:
            item_id = _item_id_for(record)
            if item_id:
                item_ids.append(item_id)

    if not item_ids:
        ap.error("nothing to do: pass --items, --refresh-existing or --catalogue")

    print(f"Ingesting {len(item_ids)} item(s) from the BIS mirror...\n")
    ingested = ingest_items(item_ids, delay=args.delay, use_cache=not args.no_cache)

    print(f"\nExtracted {len(ingested)}/{len(item_ids)} records.")

    snapshot = corpus_mod.load_snapshot(SNAPSHOT_PATH)
    delta = corpus_mod.diff_against_snapshot(ingested, snapshot)
    print(f"Change detection vs snapshot: "
          f"+{len(delta['added'])} added, ~{len(delta['changed'])} changed, "
          f"-{len(delta['removed'])} removed")
    for rid in delta["changed"][:10]:
        print(f"   changed: {rid}")

    merged, stats = corpus_mod.merge_corpus(existing, ingested)
    print(f"Merge: {stats['added']} new, {stats['updated']} refreshed, "
          f"{stats['untouched']} untouched -> {len(merged)} total")

    if args.write:
        data["standards"] = merged
        data.setdefault("_meta", {})["last_ingest"] = datetime.now(timezone.utc).isoformat()
        STANDARDS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        corpus_mod.save_snapshot(SNAPSHOT_PATH, ingested,
                                 datetime.now(timezone.utc).isoformat())
        print(f"\nWrote {STANDARDS_PATH}\nWrote {SNAPSHOT_PATH}")
    else:
        print("\n(dry run — pass --write to persist)")


if __name__ == "__main__":
    main()
