"""Merge fetched standards into the served corpus, and detect what changed.

A standards engine that is correct on the day it ships is wrong six months
later: BIS publishes revisions, amendments and withdrawals continuously. So
ingestion keeps a snapshot keyed by content hash and reports added / changed /
removed records on each run, rather than silently overwriting.

Merge policy is conservative. Hand-curated records carry fields the mirror has
no equivalent for (QCO mappings, aliases, supersession chains), so ingested
data fills gaps and refreshes document-derived fields, and never clobbers
curation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fields that come from the document itself and may be refreshed on re-ingest.
_DOCUMENT_FIELDS = ("title", "scope", "committee", "ics", "chunks",
                    "referred_standards", "source")

# Fields that are curated by hand and must survive ingestion untouched.
_CURATED_FIELDS = ("qco", "aliases", "designations", "allied", "status",
                   "superseded_by", "supersedes", "amendments")


def content_hash(record: Dict[str, Any]) -> str:
    payload = json.dumps(
        {k: record.get(k) for k in _DOCUMENT_FIELDS},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_snapshot(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("hashes", {})


def save_snapshot(path: Path, records: List[Dict[str, Any]], fetched_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fetched_at": fetched_at,
        "count": len(records),
        "hashes": {r["id"]: content_hash(r) for r in records},
    }, indent=2), encoding="utf-8")


def diff_against_snapshot(records: List[Dict[str, Any]],
                          snapshot: Dict[str, str]) -> Dict[str, List[str]]:
    current = {r["id"]: content_hash(r) for r in records}
    added = [rid for rid in current if rid not in snapshot]
    changed = [rid for rid in current if rid in snapshot and snapshot[rid] != current[rid]]
    removed = [rid for rid in snapshot if rid not in current]
    return {"added": sorted(added), "changed": sorted(changed), "removed": sorted(removed)}


def merge_record(existing: Optional[Dict[str, Any]],
                 ingested: Dict[str, Any]) -> Dict[str, Any]:
    """Ingested document data refreshes document fields; curation is preserved."""
    if existing is None:
        merged = dict(ingested)
        merged.setdefault("status", "active")
        merged.setdefault("aliases", [])
        merged.setdefault("designations", [])
        merged.setdefault("allied", [])
        merged.setdefault("amendments", [])
        merged.setdefault("qco", None)
        merged.setdefault("superseded_by", None)
        merged.setdefault("supersedes", None)
        merged.setdefault("department", ingested.get("department") or "Unclassified")
        return merged

    merged = dict(existing)
    for field in _DOCUMENT_FIELDS:
        value = ingested.get(field)
        if value:  # never blank out curated content with a failed extraction
            merged[field] = value
    for field in _CURATED_FIELDS:
        if field in existing:
            merged[field] = existing[field]

    # Verified against the real document -- upgrade confidence unless extraction
    # came back thin.
    if ingested.get("scope"):
        merged["data_confidence"] = "high"
    return merged


def merge_corpus(existing: List[Dict[str, Any]],
                 ingested: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    by_id = {r["id"]: r for r in existing}
    stats = {"updated": 0, "added": 0, "untouched": 0}

    for record in ingested:
        rid = record["id"]
        if rid in by_id:
            by_id[rid] = merge_record(by_id[rid], record)
            stats["updated"] += 1
        else:
            by_id[rid] = merge_record(None, record)
            stats["added"] += 1

    ingested_ids = {r["id"] for r in ingested}
    stats["untouched"] = sum(1 for r in existing if r["id"] not in ingested_ids)

    # Preserve original ordering, append new records after it.
    ordered = [by_id[r["id"]] for r in existing]
    ordered += [by_id[rid] for rid in (r["id"] for r in ingested) if rid not in {r["id"] for r in existing}]
    return ordered, stats
