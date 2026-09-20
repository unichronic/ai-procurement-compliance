"""Fetching from the public BIS mirror.

law.resource.org hosts a sectional index of Indian Standards that links to
archive.org items, each of which carries the full text as a .txt alongside the
scanned PDF. That plain text is what lets us index document *content* (scope
clauses, normative reference lists) instead of two sentences of metadata.

The archive.org item id encodes the citation:
    gov.in.is.10500.2012      -> IS 10500:2012
    gov.in.is.9873.4.2017     -> IS 9873 (Part 4):2017
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional

MIRROR_ROOT = "https://law.resource.org/pub/in/bis"
ARCHIVE_DOWNLOAD = "https://archive.org/download"

# Sectional divisions used by the mirror (BIS division codes).
SECTIONS = [f"S{n:02d}" for n in range(1, 15)]

_ITEM_RE = re.compile(r"archive\.org/details/(gov\.in\.is\.[0-9.]+)", re.I)
_ID_PARTS_RE = re.compile(r"^gov\.in\.is\.(?P<nums>[0-9.]+)$", re.I)

_USER_AGENT = "sih26108-standards-ingest/0.1 (hackathon prototype; contact via repo)"


def parse_catalogue_html(html: str) -> List[str]:
    """Extract unique archive.org item ids from a sectional index page."""
    seen = []
    for match in _ITEM_RE.finditer(html):
        item = match.group(1).lower().rstrip(".")
        if item not in seen:
            seen.append(item)
    return seen


def parse_item_id(item_id: str) -> Optional[Dict[str, object]]:
    """gov.in.is.9873.4.2017 -> {number: '9873', part: '4', year: 2017}.

    The trailing component is the edition year; anything between the number and
    the year is a part (and, rarely, a section).
    """
    m = _ID_PARTS_RE.match(item_id.strip().lower())
    if not m:
        return None

    parts = [p for p in m.group("nums").split(".") if p]
    if len(parts) < 2:
        return None

    year_token = parts[-1]
    if not re.fullmatch(r"(?:18|19|20)\d{2}", year_token):
        return None

    number = parts[0]
    middle = parts[1:-1]
    return {
        "item_id": item_id,
        "number": number,
        "part": middle[0] if len(middle) >= 1 else None,
        "section": middle[1] if len(middle) >= 2 else None,
        "year": int(year_token),
    }


def text_url(item_id: str) -> str:
    """archive.org stores the full text as <item-without-gov.in.>.txt"""
    stem = item_id[len("gov.in."):] if item_id.startswith("gov.in.") else item_id
    return f"{ARCHIVE_DOWNLOAD}/{item_id}/{stem}.txt"


def _get(url: str, timeout: int = 60) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def fetch_section_index(section: str, timeout: int = 60) -> List[str]:
    html = _get(f"{MIRROR_ROOT}/{section}/", timeout=timeout)
    return parse_catalogue_html(html) if html else []


def fetch_catalogue(sections: Iterable[str] = SECTIONS, delay: float = 1.0,
                    timeout: int = 60) -> List[Dict[str, object]]:
    """Walk sectional indexes and return parsed catalogue entries."""
    out: List[Dict[str, object]] = []
    seen = set()
    for section in sections:
        for item_id in fetch_section_index(section, timeout=timeout):
            if item_id in seen:
                continue
            seen.add(item_id)
            parsed = parse_item_id(item_id)
            if parsed:
                parsed["section"] = parsed.get("section")
                parsed["division"] = section
                out.append(parsed)
        time.sleep(delay)  # be polite to a free public mirror
    return out


def fetch_standard_text(item_id: str, timeout: int = 90) -> Optional[str]:
    return _get(text_url(item_id), timeout=timeout)
