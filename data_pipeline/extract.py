"""Turn raw BIS full text into structured fields and retrieval chunks.

BIS documents follow a stable layout:

    IS 10500 : 2012
    Indian Standard DRINKING WATER -- SPECIFICATION ( Second Revision )
    ICS 13.060.20
    Drinking Water Sectional Committee, FAD 25
    ...
    1 SCOPE  This standard prescribes ...
    2 REFERENCES  The standards listed in Annex A ...
    ANNEX A  (Clause 2)  LIST OF REFERRED INDIAN STANDARDS  IS 3025 ...

which means scope clauses and normative reference lists can be lifted from the
real document rather than hand-authored. That matters twice over: it populates
`chunks` for content-level retrieval, and it lets the allied-standards graph be
derived from what each standard actually cites.

Pure functions only -- no network -- so this is testable offline.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_TITLE_RE = re.compile(r"Indian\s+Standard\s+(.+?)(?:\s*\(\s*(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh)\s+Revision\s*\))?\s*(?:\n|ICS|$)", re.I | re.S)
_ICS_RE = re.compile(r"\bICS\s+([\d.,;\s]+)", re.I)
_COMMITTEE_RE = re.compile(r"Sectional\s+Committee\s*,?\s*([A-Z]{2,4}\s*\d{1,3})", re.I)
_SCOPE_RE = re.compile(r"\b1\s+SCOPE\b(.*?)(?=\b\d+\s+[A-Z]{3,}|\Z)", re.S)
_ANNEX_REF_RE = re.compile(r"LIST\s+OF\s+REFERRED\s+INDIAN\s+STANDARDS(.*?)(?=\bANNEX\s+[B-Z]\b|\Z)", re.I | re.S)
_IS_NUMBER_RE = re.compile(
    r"\bIS\s*(?:/\s*(?:IEC|ISO))?\s*:?\s*(\d{2,5})"
    r"(?:\s*\(\s*Part\s*(\d+)[^)]*\))?"
    r"(?:\s*:\s*((?:18|19|20)\d{2}))?",
    re.I,
)

# One alternation so the annex is scanned as a single left-to-right token
# stream and "(Part N)" entries stay attached to the base number before them.
_ANNEX_TOKEN_RE = re.compile(
    r"\(\s*Part\s*(?P<pnum>\d+)[^)]*\)\s*:\s*(?P<pyear>(?:18|19|20)\d{2})"
    r"|\b(?P<dated>\d{3,5})\s*:\s*(?P<dyear>(?:18|19|20)\d{2})"
    r"|\b(?P<bare>\d{3,5})\b",
    re.I,
)

_MAX_CHUNK_CHARS = 1200


def _clean(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _plausible_title(title: str) -> bool:
    """Scanned text yields junk titles ("~", "( Reaffirmed 2002 )") often enough
    that unguarded extraction would overwrite good curated data with garbage.
    Refusing a bad value is always better than propagating one."""
    if len(title) < 8:
        return False
    if title.lstrip().startswith(("(", "~", "-", "[")):
        return False
    words = re.findall(r"[A-Za-z]{2,}", title)
    if len(words) < 2:
        return False
    if re.fullmatch(r"\(?\s*Reaffirmed[^)]*\)?", title, re.I):
        return False
    return True


# BIS prints titles in caps. Naive .title() would mangle these into "Lpg",
# "Pvc", "Led", so they are passed through untouched.
_ACRONYMS = {
    "IS", "BIS", "LPG", "LED", "PVC", "XLPE", "IT", "ISI", "CRS", "HUID",
    "UV", "PH", "TV", "AC", "DC", "IEC", "ISO", "RCC", "MS", "GI", "ICS",
    "CFL", "LPGE", "HDPE", "LDPE", "PPE", "UPS", "EMI", "EMC",
}
_LOWERCASE_WORDS = {"and", "or", "of", "for", "in", "on", "to", "the", "a",
                    "an", "with", "from", "by", "at", "as"}


def _normalize_title_case(title: str) -> str:
    """Caps -> Title Case, preserving acronyms and lowercase connectives."""
    if not title.isupper():
        return title

    words = title.split()
    out = []
    for i, word in enumerate(words):
        core = re.sub(r"[^A-Za-z]", "", word)
        if core.upper() in _ACRONYMS:
            out.append(word.upper())
        elif i > 0 and core.lower() in _LOWERCASE_WORDS:
            out.append(word.lower())
        else:
            out.append(word.capitalize())
    return " ".join(out)


def extract_title(raw: str) -> Optional[str]:
    m = _TITLE_RE.search(raw)
    if not m:
        return None
    title = _clean(m.group(1))
    title = re.sub(r"\s*--\s*", " — ", title)
    if not title or not _plausible_title(title):
        return None
    return _normalize_title_case(title)


def extract_ics(raw: str) -> Optional[str]:
    m = _ICS_RE.search(raw)
    return _clean(m.group(1)).rstrip(",;") if m else None


def extract_committee(raw: str) -> Optional[str]:
    m = _COMMITTEE_RE.search(raw)
    return re.sub(r"\s+", " ", m.group(1)).strip().upper() if m else None


_MIN_SCOPE_CHARS = 30


def extract_scope(raw: str) -> Optional[str]:
    m = _SCOPE_RE.search(raw)
    if not m:
        return None
    scope = _clean(m.group(1))[:_MAX_CHUNK_CHARS]
    # A scope clause shorter than a sentence is an extraction failure, not a
    # terse standard.
    return scope if len(scope) >= _MIN_SCOPE_CHARS else None


def extract_referred_standards(raw: str, *, self_number: Optional[str] = None) -> List[Dict[str, Any]]:
    """Normative references, from the Annex that lists them.

    The annex is a two-column table ("IS No." | "Title") and the scanned text
    flattens it, so the numbers arrive as a stream rather than beside their
    titles:

        IS No. 1622 : 1981  Title Methods of sampling ... 15302 : 2002
        ... 13428 : 2003  3025  (Part 1) : 1987 (Part 2) : 2002 ...

    A bare number establishes a base that subsequent "(Part N) : year" entries
    attach to -- that's how IS 3025's forty-odd parts are listed. Page headers
    repeat the document's own number throughout, so those are filtered out.
    """
    m = _ANNEX_REF_RE.search(raw)
    block = m.group(1) if m else ""
    if not block:
        return []

    out: List[Dict[str, Any]] = []
    seen = set()
    current_base: Optional[str] = None

    def add(number: str, part: Optional[str], year: Optional[int]):
        if self_number and number == str(self_number):
            return  # running page header, not a reference
        key = (number, part, year)
        if key in seen:
            return
        seen.add(key)
        out.append({"number": number, "part": part, "year": year})

    for token in _ANNEX_TOKEN_RE.finditer(block):
        if token.group("pnum"):           # "(Part 4) : 1994" under a base
            # Only attach to a base established by a BARE number, i.e. a column
            # header introducing a part list. A dated entry ("15303 : 2002") is
            # a complete citation in its own right, and a part run following it
            # usually belongs to a base further down that the page break
            # separated. Attaching them anyway invents normative references,
            # and a wrong reference produces a wrong lint finding -- so these
            # are dropped instead. Missing beats fabricated.
            if current_base is not None:
                add(current_base, token.group("pnum"), int(token.group("pyear")))
        elif token.group("dated"):        # "15302 : 2002" -- complete citation
            add(token.group("dated"), None, int(token.group("dyear")))
            current_base = None
        elif token.group("bare"):         # "3025" heading a run of parts
            current_base = token.group("bare")

    return out


def build_chunks(raw: str, *, self_number: Optional[str] = None) -> List[Dict[str, str]]:
    """Retrieval chunks. Scope is the single most discriminative passage in a
    BIS document, so it is always its own chunk when present."""
    chunks: List[Dict[str, str]] = []

    scope = extract_scope(raw)
    if scope:
        chunks.append({"kind": "scope", "text": scope})

    refs = extract_referred_standards(raw, self_number=self_number)
    if refs:
        rendered = ", ".join(
            f"IS {r['number']}" + (f" (Part {r['part']})" if r["part"] else "")
            for r in refs
        )
        chunks.append({"kind": "normative_references", "text": f"Refers to: {rendered}"})

    chunks.extend(_body_chunks(raw))
    return chunks


_BOILERPLATE_RE = re.compile(
    r"BUREAU OF INDIAN STANDARDS|MANAK BHAVAN|BAHADUR SHAH ZAFAR|"
    r"Price Group|FOREWORD|© BIS|Printed at", re.I)

_MAX_BODY_CHUNKS = 8


def _body_chunks(raw: str) -> List[Dict[str, str]]:
    """Passages from the requirements body.

    This is the point of ingesting full text at all: a standard's
    discriminating content (grades, dimensions, test conditions, marking
    clauses) lives in numbered requirement clauses, not in the two-sentence
    scope. Title pages, foreword and colophon carry no retrieval signal and are
    dropped.
    """
    body_start = raw.find("1 SCOPE")
    body = raw[body_start:] if body_start > 0 else raw
    body = body.replace("\x0c", " ")

    # Split on numbered clause headings ("4 REQUIREMENTS", "5.2 Marking").
    parts = re.split(r"(?=\b\d+(?:\.\d+)*\s+[A-Z][A-Za-z ]{3,})", body)

    out: List[Dict[str, str]] = []
    for part in parts:
        text = _clean(part)
        if len(text) < 120 or _BOILERPLATE_RE.search(text):
            continue
        out.append({"kind": "body", "text": text[:_MAX_CHUNK_CHARS]})
        if len(out) >= _MAX_BODY_CHUNKS:
            break
    return out


def extract_standard(raw: str, *, item: Dict[str, Any]) -> Dict[str, Any]:
    """Full extraction for one standard, keyed off its catalogue entry."""
    number = f"IS {item['number']}"
    if item.get("part"):
        number += f" (Part {item['part']})"

    return {
        "id": _make_id(item),
        "number": number,
        "edition_year": item.get("year"),
        "title": extract_title(raw),
        "scope": extract_scope(raw),
        "committee": extract_committee(raw),
        "ics": extract_ics(raw),
        "referred_standards": extract_referred_standards(raw, self_number=item["number"]),
        "chunks": build_chunks(raw, self_number=item["number"]),
        "source": {
            "item_id": item.get("item_id"),
            "mirror": "law.resource.org/pub/in/bis",
        },
        "data_confidence": "high" if extract_scope(raw) else "low",
    }


def _make_id(item: Dict[str, Any]) -> str:
    bits = ["IS", str(item["number"])]
    if item.get("part"):
        bits.append(str(item["part"]))
    if item.get("section"):
        bits.append(str(item["section"]))
    if item.get("year"):
        bits.append(str(item["year"]))
    return "_".join(bits)


_MOJIBAKE_MARKERS = ("Ã", "â€", "Â", "�")


def _mojibake_damage(text: str) -> int:
    return sum(text.count(m) for m in _MOJIBAKE_MARKERS)


def repair_mojibake(text: str, max_rounds: int = 3) -> str:
    """Undo UTF-8 that was decoded as cp1252, possibly more than once.

    Archive titles arrive with damage like "BullÃ¢â‚¬â„¢s Trench" for "Bull's
    Trench" — UTF-8 bytes read as cp1252, then re-encoded and read as cp1252
    again. So the repair iterates rather than assuming a single round.

    cp1252, not latin-1: the damaged text contains characters like U+20AC (€)
    and U+201A (‚) that only exist in the Windows codepage, so a latin-1
    encode raises and the repair silently does nothing — which is exactly what
    the first version of this did.

    Each round is kept only if it strictly reduces damage, so legitimate text
    containing 'Ã' is never mangled by a speculative re-decode.
    """
    if not text or not any(m in text for m in _MOJIBAKE_MARKERS):
        return text

    current = text
    for _ in range(max_rounds):
        best = current
        for encode in (_encode_cp1252_mixed, _encode_strict_cp1252, _encode_latin1):
            try:
                candidate = encode(current).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if _mojibake_damage(candidate) < _mojibake_damage(best):
                best = candidate
        if best == current:
            break
        current = best
    return current


def _encode_strict_cp1252(text: str) -> bytes:
    return text.encode("cp1252")


def _encode_latin1(text: str) -> bytes:
    return text.encode("latin-1")


def _encode_cp1252_mixed(text: str) -> bytes:
    """Encode cp1252, treating C1 controls as their raw byte.

    The worst damage mixes cp1252-only characters (U+201A, U+20AC) with
    U+009D, which cp1252 leaves unassigned — so a strict cp1252 encode raises
    and a latin-1 encode raises on the others. Neither codec can do the whole
    string, but each character individually is representable, so this encodes
    per character and falls back to the codepoint for the C1 range.
    """
    out = bytearray()
    for ch in text:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            code = ord(ch)
            if 0x80 <= code <= 0x9F:      # C1 control, unassigned in cp1252
                out.append(code)
            else:
                raise
    return bytes(out)


_CLAUSE2_RE = re.compile(r"\b2\s+REFERENCES\b(.{0,2500}?)(?=\b\d+\s+[A-Z]{3,}|\Z)", re.S)


def extract_clause2_references(raw: str, *, self_number: Optional[str] = None) -> List[Dict[str, Any]]:
    """Normative references listed inline in clause 2, rather than in an annex.

    Only about one document in forty carries a "LIST OF REFERRED INDIAN
    STANDARDS" annex; many instead name the standards directly in clause 2,
    which is normative by definition.

    Restricted to clause 2 on purpose. Harvesting IS numbers from anywhere in
    the body would find passing mentions as readily as dependencies, and a
    fabricated normative reference becomes a wrong lint finding told to a
    procurement officer.
    """
    m = _CLAUSE2_RE.search(raw)
    if not m:
        return []

    out: List[Dict[str, Any]] = []
    seen = set()
    for match in _IS_NUMBER_RE.finditer(m.group(1)):
        number, part, year = match.group(1), match.group(2), match.group(3)
        if self_number and number == str(self_number):
            continue
        key = (number, part)
        if key in seen:
            continue
        seen.add(key)
        out.append({"number": number, "part": part,
                    "year": int(year) if year else None})
    return out
