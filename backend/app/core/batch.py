"""
Splits a whole tender/spec document into individual line items so each
one can be scored independently (PS requirement: accept "tender documents"
as input, not just a single short query).

Spans are tracked because the linter reports findings against a position in
the officer's own document ("line 4, here") rather than in the abstract -- a
finding you can't locate is a finding nobody acts on.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•]|\(?[a-zA-Z0-9]{1,3}[\).:]|\d+\.)\s*")


def split_document_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """-> [(cleaned_line, start_offset, end_offset)] into the original text."""
    items: List[Tuple[str, int, int]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(raw_line)

        stripped = raw_line.strip()
        if not stripped:
            continue

        cleaned = _BULLET_PREFIX.sub("", stripped).strip()
        if len(cleaned) < 4:
            continue

        # Locate the cleaned text inside the original line so the span points at
        # the content, not the bullet marker.
        idx = raw_line.find(cleaned)
        start = line_start + (idx if idx >= 0 else 0)
        items.append((cleaned, start, start + len(cleaned)))

    return items


def split_document(text: str) -> List[str]:
    return [item for item, _, _ in split_document_with_spans(text)]
