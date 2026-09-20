"""Audit a draft tender specification for standards defects.

This is the inversion of /recommend. Search answers "which standard applies?"
for an officer who already knows to ask. The PS Background describes the
opposite situation: specs that "omit relevant standards, reference outdated
versions, or include incomplete technical requirements" -- defects in a draft
nobody is currently checking. So this takes a whole draft and reports what is
wrong with it.

Every rule here is deterministic and data-driven. No LLM decides whether a
finding fires: a finding carries legal and procurement consequences, so it has
to be reproducible and traceable to the record that produced it. The LLM's only
role is narrating a finding after the fact (see explain.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.batch import split_document_with_spans
from app.core.citation import (
    CitationResolver,
    format_citation,
    has_marking_requirement,
    parse_citations,
)

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Allied relation types a compliant spec is normally expected to cite alongside
# the product standard. Terminology/related_product are informational, so they
# don't raise findings.
EXPECTED_ROLES = ("normative_reference", "test_method")

ROLE_LABELS = {
    "normative_reference": "normative reference",
    "test_method": "test method",
}


def _finding(rule_id, severity, message, *, span=None, evidence=None,
             suggested_fix=None, authority=None, standard_id=None) -> Dict[str, Any]:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "message": message,
        "span": span,
        "evidence": evidence,
        "suggested_fix": suggested_fix,
        "authority": authority,
        "standard_id": standard_id,
    }


class SpecLinter:
    def __init__(self, standards, index, versions, allied, certification):
        self.standards = standards
        self.index = index
        self.versions = versions
        self.allied = allied
        self.certification = certification
        self.resolver = CitationResolver(standards)

    def lint(self, text: str, *, suggest_missing: bool = True) -> Dict[str, Any]:
        citations = parse_citations(text)
        resolved: List[Dict[str, Any]] = []
        findings: List[Dict[str, Any]] = []

        for c in citations:
            standard = self.resolver.resolve(c)
            resolved.append({**c, "standard_id": standard["id"] if standard else None})
            if standard is None:
                findings.append(_finding(
                    "unresolved_citation", "medium",
                    f"Citation '{c['raw']}' does not match any standard in the corpus. "
                    f"It may be mistyped, withdrawn, or outside the indexed scope.",
                    span=c["span"], evidence=c["raw"],
                    suggested_fix="Verify the number against the BIS catalogue.",
                ))

        cited_ids = {r["standard_id"] for r in resolved if r["standard_id"]}

        findings += self._rule_superseded(resolved)
        findings += self._rule_missing_allied(resolved, cited_ids)
        findings += self._rule_missing_marking(resolved, text)
        if suggest_missing:
            findings += self._rule_uncited_items(text, cited_ids)

        findings.sort(key=lambda f: (SEVERITY_ORDER.get(f["severity"], 9),
                                     f["span"][0] if f["span"] else 10**9))

        return {
            "citations_found": len(citations),
            "citations_resolved": len(cited_ids),
            "findings": findings,
            "summary": self._summarize(findings),
        }

    # -- rules ---------------------------------------------------------------

    def _rule_superseded(self, resolved) -> List[Dict[str, Any]]:
        """Cited edition is no longer the current one."""
        out = []
        for r in resolved:
            if not r["standard_id"]:
                continue
            info = self.versions.resolve(r["standard_id"])
            if info.get("is_current"):
                continue

            latest = info.get("latest_active") or []
            if latest:
                target = next(
                    (s for s in self.standards if s["id"] == latest[0]["id"]), None
                )
                fix = (f"Cite {format_citation(target)}" if target
                       else f"Cite {latest[0]['number']}")
            else:
                fix = "No active successor found in the corpus; verify with BIS."

            out.append(_finding(
                "superseded_citation", "high",
                f"{r['raw']} refers to a {info['status'].upper()} edition. "
                f"A tender citing it can be challenged as referencing an "
                f"outdated specification.",
                span=r["span"], evidence=r["raw"], suggested_fix=fix,
                authority="BIS catalogue supersession record",
                standard_id=r["standard_id"],
            ))
        return out

    def _rule_missing_allied(self, resolved, cited_ids) -> List[Dict[str, Any]]:
        """A cited standard normatively depends on standards the draft omits."""
        out = []
        seen = set()
        for r in resolved:
            sid = r["standard_id"]
            if not sid or sid in seen:
                continue
            seen.add(sid)

            grouped = self.allied.grouped_allied(sid)
            for role in EXPECTED_ROLES:
                for edge in grouped.get(role, []):
                    if edge["direction"] != "forward" or edge["id"] in cited_ids:
                        continue
                    out.append(_finding(
                        "missing_allied_standard", "medium",
                        f"{r['raw']} relies on {edge['number']} "
                        f"({ROLE_LABELS[role]}) but the draft does not cite it. "
                        f"{edge['relation_note']}",
                        span=r["span"], evidence=r["raw"],
                        suggested_fix=f"Add a reference to {edge['number']} — {edge['title']}.",
                        authority=f"Allied standard relation: {role}",
                        standard_id=edge["id"],
                    ))
        return out

    def _rule_missing_marking(self, resolved, text) -> List[Dict[str, Any]]:
        """Cited product falls under a QCO but the draft never demands the mark."""
        if has_marking_requirement(text):
            return []

        out = []
        seen = set()
        for r in resolved:
            sid = r["standard_id"]
            if not sid or sid in seen:
                continue
            seen.add(sid)

            cert = self.certification.advise(sid)
            if not cert.get("mandatory_certification"):
                continue

            order = cert.get("governing_order") or {}
            out.append(_finding(
                "missing_certification_requirement", "high",
                f"Products under {r['raw']} require mandatory {cert['scheme']} "
                f"certification, but the draft does not require a conformity "
                f"mark anywhere. A supplier could deliver uncertified goods and "
                f"remain contractually compliant.",
                span=r["span"], evidence=r["raw"],
                suggested_fix=(
                    f"Add: \"The product shall bear a valid {cert['scheme']} mark "
                    f"under {order.get('name', 'the applicable Quality Control Order')}.\""
                ),
                authority=cert.get("message"),
                standard_id=sid,
            ))
        return out

    def _rule_uncited_items(self, text, cited_ids) -> List[Dict[str, Any]]:
        """A line item describes a product but cites no standard at all."""
        out = []
        for line, start, end in split_document_with_spans(text):
            if parse_citations(line):
                continue

            hits = self.index.search(line, top_k=1)
            if not hits:
                continue
            top = hits[0]
            if top["semantic_similarity"] < 0.45:
                continue  # too weak to assert anything

            s = top["standard"]
            if s["id"] in cited_ids:
                continue

            out.append(_finding(
                "uncited_item", "medium",
                f"This item does not cite any Indian Standard. "
                f"{s['number']} appears applicable.",
                span=[start, end], evidence=line,
                suggested_fix=f"Consider citing {format_citation(s)} — {s['title']}.",
                authority=f"Semantic match, similarity {top['semantic_similarity']:.2f}",
                standard_id=s["id"],
            ))
        return out

    @staticmethod
    def _summarize(findings) -> Dict[str, int]:
        summary = {"high": 0, "medium": 0, "low": 0, "total": len(findings)}
        for f in findings:
            summary[f["severity"]] = summary.get(f["severity"], 0) + 1
        return summary
