"""
Allied/cross-referenced/normative standards, split into the six categories
the PS text names explicitly: normative_reference, test_method,
terminology, safety, installation, related_product.

Builds a bidirectional graph at load time: if A lists B as an allied
standard of type T, a query landing on B will still surface A (labelled
as the inverse relation), since a procurement officer searching by the
component standard should still be pointed back to the parent spec.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

ALLIED_TYPES = [
    "normative_reference",
    "test_method",
    "terminology",
    "safety",
    "installation",
    "related_product",
]


class AlliedGraph:
    def __init__(self, standards: List[Dict[str, Any]]):
        self.by_id = {s["id"]: s for s in standards}
        # forward[std_id] = list of (target_id, type, note)
        self.forward: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        # inverse[std_id] = list of (source_id, inverse_type, note)
        self.inverse: Dict[str, List[Dict[str, str]]] = defaultdict(list)

        for s in standards:
            for edge in s.get("allied", []):
                target = edge["id"]
                rel_type = edge["type"]
                note = edge.get("note", "")
                self.forward[s["id"]].append({"id": target, "type": rel_type, "note": note})
                self.inverse[target].append({"id": s["id"], "type": rel_type, "note": note})

    def grouped_allied(self, standard_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Returns the six PS-named categories (always present, empty list if none),
        each entry resolved to a short standard summary, not just an id."""
        grouped: Dict[str, List[Dict[str, Any]]] = {t: [] for t in ALLIED_TYPES}

        for edge in self.forward.get(standard_id, []):
            target = self.by_id.get(edge["id"])
            if not target:
                continue
            grouped.setdefault(edge["type"], []).append({
                "id": target["id"],
                "number": target["number"],
                "title": target["title"],
                "status": target["status"],
                "relation_note": edge["note"],
                "direction": "forward",
            })

        for edge in self.inverse.get(standard_id, []):
            source = self.by_id.get(edge["id"])
            if not source:
                continue
            grouped.setdefault(edge["type"], []).append({
                "id": source["id"],
                "number": source["number"],
                "title": source["title"],
                "status": source["status"],
                "relation_note": edge["note"],
                "direction": "inverse",
            })

        return grouped
