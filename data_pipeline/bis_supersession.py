"""Ingest authoritative supersession from the BIS standards-review service.

`derive_supersession.py` infers that within one IS number the newest edition
supersedes the older ones. That is sound but limited: it cannot express
cross-number supersession (IS 226 was superseded by IS 2062, IS 10601 by
IS/IEC TR 62271-301), which is the case an officer is most likely to get
wrong, and it cannot distinguish "withdrawn" from "superseded".

BIS publishes the real thing, per technical committee, with no login:

    services.bis.gov.in/php/BIS_2.0/bisconnect/standard_review/Standard_review/
        all_withdrawn_stndrd_commtt?commttid=<base64 id>&commttname=<base64>

returning a table of  S.No. | IS Number | IS Title | Category | Superseded By.

This is the authority that replaces an inference, so records sourced here are
marked `supersession_source: bis_standards_review` and reported by the linter
at high severity, while inferred ones stay at medium.

    python3 -m data_pipeline.bis_supersession --scan 1 400 --write
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
STANDARDS = ROOT / "backend" / "app" / "data" / "standards.json"
CACHE_DIR = ROOT / "data_pipeline" / "cache" / "bis_review"

BASE_URL = (
    "https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/standard_review/"
    "Standard_review/all_withdrawn_stndrd_commtt"
)
_USER_AGENT = "sih26108-standards-ingest/0.1 (hackathon prototype)"

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")

# "IS 12661 : Part 1 : 1988", "IS/IEC TR 62271-301 : 2009", "IS 10601 : 1983"
_NUM_RE = re.compile(
    r"IS(?:/(?:IEC|ISO))?(?:\s+TR)?\s*(?P<base>\d{2,5})"
    r"(?:\s*[-–]\s*(?P<dashpart>\d{1,3}))?"
    r"(?:\s*:\s*Part\s*(?P<part>\d+))?"
    r"(?:\s*:\s*(?P<year>(?:18|19|20)\d{2}))?",
    re.I,
)

_PLACEHOLDER = re.compile(r"^[-—\s]*$")


def _clean_cell(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub("", html)).replace("&nbsp;", " ").strip()


def parse_number(text: str) -> Optional[Dict[str, Any]]:
    m = _NUM_RE.search(text or "")
    if not m:
        return None
    part = m.group("part") or m.group("dashpart")
    return {
        "base": m.group("base"),
        "part": part,
        "year": int(m.group("year")) if m.group("year") else None,
    }


def parse_review_table(html: str) -> List[Dict[str, Any]]:
    """-> [{number, title, category, superseded_by}] for real data rows."""
    out = []
    for row in _ROW_RE.findall(html):
        cells = [_clean_cell(c) for c in _CELL_RE.findall(row)]
        cells = [c for c in cells if c != ""]
        if len(cells) < 4:
            continue
        if cells[0].lower().startswith("s.no"):
            continue
        # S.No. | IS Number | Title | Category | Superseded By
        if not cells[0].rstrip(".").isdigit():
            continue
        number = cells[1]
        superseded_by = cells[4] if len(cells) > 4 else ""
        out.append({
            "number": number,
            "title": cells[2],
            "category": cells[3],
            "superseded_by": "" if _PLACEHOLDER.match(superseded_by) else superseded_by,
        })
    return out


def fetch_committee(committee_id: int, *, timeout: int = 45,
                    use_cache: bool = True) -> Optional[str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{committee_id}.html"
    if use_cache and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")

    cid = base64.b64encode(str(committee_id).encode()).decode()
    url = f"{BASE_URL}?commttid={urllib.parse.quote(cid)}&commttname="
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    cached.write_text(html, encoding="utf-8")
    return html


def corpus_index(standards: List[Dict[str, Any]]):
    """(base, part) -> records, newest first, so a number with no year in the
    BIS table can still be matched to the editions we hold."""
    from collections import defaultdict
    idx = defaultdict(list)
    for s in standards:
        p = parse_number(s.get("number", ""))
        if not p:
            continue
        idx[(p["base"], p["part"] or "")].append(s)
    for key in idx:
        idx[key].sort(key=lambda r: r.get("edition_year") or 0, reverse=True)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs=2, type=int, metavar=("FROM", "TO"),
                    default=[1, 400], help="committee id range to fetch")
    ap.add_argument("--delay", type=float, default=0.6,
                    help="pause between requests; BIS is a public service")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    doc = json.loads(STANDARDS.read_text(encoding="utf-8"))
    standards = doc["standards"]
    idx = corpus_index(standards)

    rows: List[Dict[str, Any]] = []
    committees_with_data = 0
    for cid in range(args.scan[0], args.scan[1] + 1):
        html = fetch_committee(cid, use_cache=not args.no_cache)
        if not html:
            continue
        parsed = parse_review_table(html)
        if parsed:
            committees_with_data += 1
            rows.extend(parsed)
        time.sleep(args.delay)

    print(f"committees with data : {committees_with_data}")
    print(f"withdrawn/superseded rows: {len(rows)}")
    with_successor = [r for r in rows if r["superseded_by"]]
    print(f"  of which name a successor: {len(with_successor)}")

    applied = cross_number = withdrawn_only = unmatched = ambiguous = 0
    for row in rows:
        src = parse_number(row["number"])
        if not src:
            continue
        candidates = idx.get((src["base"], src["part"] or ""))
        if not candidates:
            unmatched += 1
            continue
        if src["year"]:
            targets = [c for c in candidates if c.get("edition_year") == src["year"]]
            if not targets:
                # BIS names an edition we do not hold. Applying its withdrawal to
                # a different edition would be a guess, and the wrong guess here
                # tells an officer not to cite a standard that is in fact
                # current.
                ambiguous += 1
                continue
        elif len(candidates) == 1:
            targets = candidates
        else:
            # No year in the BIS row and several editions held: cannot tell
            # which was withdrawn. Marking them all would risk flagging the
            # current edition as withdrawn — a false high-severity finding.
            ambiguous += 1
            continue

        succ_ids: List[str] = []
        if row["superseded_by"]:
            sp = parse_number(row["superseded_by"])
            if sp:
                succ = idx.get((sp["base"], sp["part"] or ""))
                if succ:
                    pick = ([c for c in succ if c.get("edition_year") == sp["year"]]
                            if sp["year"] else succ)
                    if pick:
                        succ_ids = [pick[0]["id"]]
                        if sp["base"] != src["base"]:
                            cross_number += 1

        for target in targets:
            target["status"] = "withdrawn" if not succ_ids else "superseded"
            if succ_ids:
                target["superseded_by"] = succ_ids
            target["supersession_source"] = "bis_standards_review"
            target["bis_review_category"] = row["category"] or None
            applied += 1
            if not succ_ids:
                withdrawn_only += 1

    print(f"corpus records updated    : {applied}")
    print(f"  cross-number supersession (inference cannot produce these): {cross_number}")
    print(f"  withdrawn with no successor named: {withdrawn_only}")
    print(f"  BIS rows not in our corpus: {unmatched}")
    print(f"  skipped as ambiguous (edition not identifiable): {ambiguous}")

    if args.write:
        doc["standards"] = standards
        doc.setdefault("_meta", {})["bis_supersession"] = {
            "records_updated": applied,
            "cross_number": cross_number,
            "source": BASE_URL,
            "note": ("Authoritative withdrawal/supersession from the BIS standards "
                     "review service, replacing edition-year inference where the "
                     "two disagree."),
        }
        STANDARDS.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        print(f"\nWrote {STANDARDS}")
    else:
        print("\n(dry run — pass --write)")


if __name__ == "__main__":
    main()
