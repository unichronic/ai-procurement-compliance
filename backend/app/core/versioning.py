"""
Version/amendment awareness (PS requirement: "highlight the latest published
version and amendments"). Resolves supersession chains so a query that
naively lands on a withdrawn edition (e.g. IS 325) is redirected to the
current one, rather than silently returned as the answer.
"""
from __future__ import annotations

from typing import Any, Dict, List


class VersionResolver:
    def __init__(self, standards: List[Dict[str, Any]]):
        self.by_id = {s["id"]: s for s in standards}

    def resolve(self, standard_id: str) -> Dict[str, Any]:
        s = self.by_id.get(standard_id)
        if not s:
            return {"error": f"unknown standard id {standard_id}"}

        chain: List[str] = [standard_id]
        current = s
        # Follow supersession chain to the current, active edition.
        # (superseded_by can list multiple targets, e.g. a standard split
        # into two IE/IEC-aligned successors -- we surface all of them.)
        latest_ids: List[str] = []
        if current["status"] == "superseded" or current["status"] == "withdrawn":
            frontier = current.get("superseded_by") or []
            seen = set(chain)
            while frontier:
                nxt_id = frontier.pop(0)
                if nxt_id in seen:
                    continue
                seen.add(nxt_id)
                chain.append(nxt_id)
                nxt = self.by_id.get(nxt_id)
                if not nxt:
                    continue
                if nxt["status"] == "active":
                    latest_ids.append(nxt_id)
                else:
                    frontier.extend(nxt.get("superseded_by") or [])
        else:
            latest_ids = [standard_id]

        return {
            "queried_id": standard_id,
            "queried_number": s["number"],
            "status": s["status"],
            "is_current": s["status"] == "active",
            "supersession_chain": chain,
            "latest_active_ids": latest_ids,
            "latest_active": [
                {"id": lid, "number": self.by_id[lid]["number"], "title": self.by_id[lid]["title"]}
                for lid in latest_ids if lid in self.by_id
            ],
            "amendments": s.get("amendments", []),
            "warning": (
                f"{s['number']}:{s['edition_year']} is {s['status'].upper()}. "
                f"Cite the current edition instead."
                if s["status"] != "active" else None
            ),
        }
