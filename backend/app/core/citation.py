"""Parse and emit Indian Standard citations.

Real tender documents cite standards a dozen different ways -- "IS:2062",
"IS 2062-2011", "IS 2062 : 2011", "IS 9873(P-4):2017", "IS 9873 (Part 4)",
"IS/IEC 60034-1", "BIS 10500". Anything that audits a draft spec has to
recognise all of them before it can say a word about whether they're correct,
so this is the front door of the linter.

Also generates the citation string an officer should actually paste back into
the tender, including amendment state -- "IS 456:2000, incorporating Amendment
No. 6 (2024-06)" -- since citing a bare number loses the amendment level the
procurement is meant to conform to.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Prefix + number, then optional part/year suffixes scanned from the tail.
# The trailing group is deliberately greedy-ish but bounded: it only consumes
# bracketed part markers and 2-4 digit run-ons, never free text.
_CITATION_RE = re.compile(
    r"""\b
    (?P<prefix>IS\s*/\s*(?:IEC|ISO)|IS|BIS)      # IS, BIS, IS/IEC, IS/ISO
    \s*:?\s*
    (?P<number>\d{2,5})                           # base number
    (?P<tail>
        (?:\s*[-–]\s*\d{1,4})?                    # -1 (part) or -2011 (year)
        (?:\s*\(\s*(?:Part|Pt\.?|P)\s*[-\s]?\d+   # (Part 4) / (P-4)
           (?:\s*/\s*Sec(?:tion)?\.?\s*\d+)?      # optional /Sec 1
           \s*\))?
        (?:\s*[:\-–]\s*(?:19|20)\d{2})?           # :2011
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PART_RE = re.compile(r"(?:Part|Pt\.?|P)\s*[-\s]?(\d+)", re.IGNORECASE)
_SEC_RE = re.compile(r"Sec(?:tion)?\.?\s*(\d+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

_MARKING_KEYWORDS = (
    "isi mark", "isi marked", "isi-marked", "standard mark", "bis mark",
    "bis certified", "bis certification", "bis licence", "bis license",
    "crs registration", "crs registered", "hallmark", "hallmarked", "huid",
    "bis registration", "r-number",
)


def _normalize_number_field(number: str) -> Dict[str, Optional[str]]:
    """Corpus `number` strings ("IS 875 (Part 3)", "IS/IEC 60034-1") -> parts."""
    base_match = re.search(r"(\d{2,5})", number)
    base = base_match.group(1) if base_match else None

    part = None
    part_match = _PART_RE.search(number)
    if part_match:
        part = part_match.group(1)
    else:
        # "IS/IEC 60034-1" style: trailing -N is a part, not a year
        dash = re.search(r"\d{2,5}\s*[-–]\s*(\d{1,2})\b", number)
        if dash:
            part = dash.group(1)

    section = None
    sec_match = _SEC_RE.search(number)
    if sec_match:
        section = sec_match.group(1)

    return {"base": base, "part": part, "section": section}


def parse_citations(text: str) -> List[Dict[str, Any]]:
    """Extract every IS citation in `text`, with character spans."""
    out: List[Dict[str, Any]] = []
    for m in _CITATION_RE.finditer(text):
        tail = m.group("tail") or ""

        part = None
        part_match = _PART_RE.search(tail)
        if part_match:
            part = part_match.group(1)

        section = None
        sec_match = _SEC_RE.search(tail)
        if sec_match:
            section = sec_match.group(1)

        year = None
        year_match = _YEAR_RE.search(tail)
        if year_match:
            year = int(year_match.group(0))

        # A bare "-N" with no part marker and no year: 1-2 digits is a part
        # (IS/IEC 60034-1), 4 digits would already have matched as a year.
        if part is None and year is None:
            dash = re.match(r"\s*[-–]\s*(\d{1,2})\b", tail)
            if dash:
                part = dash.group(1)

        out.append({
            "raw": m.group(0).strip(),
            "base": m.group("number"),
            "part": part,
            "section": section,
            "year": year,
            "span": [m.start(), m.end()],
        })
    return out


class CitationResolver:
    """Maps parsed citations onto corpus standards."""

    def __init__(self, standards: List[Dict[str, Any]]):
        self.standards = standards
        self._index: Dict[str, List[Dict[str, Any]]] = {}
        for s in standards:
            key = self._key_for_standard(s)
            self._index.setdefault(key, []).append(s)

    @staticmethod
    def _key_for_standard(s: Dict[str, Any]) -> str:
        parsed = _normalize_number_field(s["number"])
        return f"{parsed['base']}|{parsed['part'] or ''}|{parsed['section'] or ''}"

    def resolve(self, citation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the standard a citation points at.

        With a year, matches that edition exactly -- important, because the
        whole point is to notice when someone cited a superseded one. Without
        a year, prefers the active edition.
        """
        key = f"{citation['base']}|{citation['part'] or ''}|{citation['section'] or ''}"
        candidates = self._index.get(key)

        if not candidates and citation["part"] is None:
            # "IS 9873" with no part: fall back to any part of that number.
            candidates = [
                s for s in self.standards
                if _normalize_number_field(s["number"])["base"] == citation["base"]
            ]
        if not candidates:
            return None

        if citation["year"] is not None:
            exact = [s for s in candidates if s.get("edition_year") == citation["year"]]
            if exact:
                return exact[0]

        active = [s for s in candidates if s.get("status") == "active"]
        pool = active or candidates
        return max(pool, key=lambda s: s.get("edition_year") or 0)


def format_citation(standard: Dict[str, Any]) -> str:
    """The string an officer should paste into the tender."""
    base = f"{standard['number']}:{standard['edition_year']}"
    amendments = standard.get("amendments") or []
    if amendments:
        latest = max(amendments, key=lambda a: a.get("no", 0))
        base += f", incorporating Amendment No. {latest['no']} ({latest.get('date', 'n.d.')})"
    return base


def has_marking_requirement(text: str) -> bool:
    """Does the draft already require a conformity mark anywhere?"""
    low = text.lower()
    return any(kw in low for kw in _MARKING_KEYWORDS)
