"""Compatibility layer for the team's React frontend.

`frontend/` was written against `standards-retrieval/main.py`, which exposes
`/retrieve`, `/languages`, `/extract` and `/standards/{id}/...`. This engine
exposes `/recommend`, `/lint` and `/standard/{id}`. Pointing the React app at
this engine therefore did nothing useful until the contracts were reconciled.

Rather than rewrite 20 React screens or serve the weaker engine, this maps
their contract onto this one. Their response shape is the public interface
here, so the field names and the `stage_scores` / `confidence` structure are
theirs, not chosen afresh.

Their `CertificationInfo` already carries a `not_verified` scheme and defines
`mandatory` as "True only when a scheme was positively confirmed", which lines
up exactly with this engine's three-state certification. Unknown maps to
`not_verified` with `mandatory: false` and an explanation that says so, never
to a clearance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

router = APIRouter()

# Confidence gate, adopted from their main.py. The UI drops the "Recommended
# standards" heading entirely on "none" and presents results as nearest text
# matches instead -- the same instinct as this engine's lint similarity floor,
# applied at the presentation layer.
#
# Calibrated against measured top-1 similarity on this corpus:
#
#   real queries      0.504 - 0.811
#   nonsense queries  0.377 - 0.529
#
# Those ranges OVERLAP, so no cosine threshold separates them cleanly. The
# thresholds are therefore set for the one property that matters -- nonsense
# must never be reported as "strong" -- and "uncertain" is left genuinely
# uncertain rather than pretending to a precision the signal does not have.
# Their gate used cross-encoder logits, which have a wider dynamic range;
# migrating to that would sharpen this.
STRONG_SIMILARITY = 0.60
NONE_SIMILARITY = 0.40


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10
    language: Optional[str] = None
    explain: bool = False


RELATION_HEADINGS = {
    "normative_reference": "Normative references",
    "test_method": "Test methods",
    "terminology": "Terminology",
    "safety": "Safety",
    "installation": "Installation and use",
    "related_product": "Related products",
}

RELATION_EXPLANATIONS = {
    "normative_reference": ("Cited in this standard's own annex; a specification "
                           "citing this standard normally cites these too."),
    "test_method": "Defines how conformity with this standard is verified.",
}


def _citation_number(s: Dict[str, Any]) -> str:
    """Designation including the edition, e.g. "IS 456:2000".

    Their contract documents `number` as the full BIS designation ("IS
    694:2010"); this engine stores designation and edition separately. Emitting
    the bare number made every detail page render "IS 456" where the UI, and
    any officer reading it, expects the citable form.
    """
    number = s.get("number", "")
    year = s.get("edition_year")
    return f"{number}:{year}" if number and year else number


def _certification_block(cert: Dict[str, Any]) -> Dict[str, Any]:
    status = cert.get("certification_status")
    scheme = cert.get("scheme")

    if cert.get("mandatory_certification") is True:
        return {
            "scheme": scheme if scheme in ("ISI", "CRS", "Hallmark") else "not_verified",
            "mandatory": True,
            "explanation": cert.get("message", ""),
            "qco": (cert.get("governing_order") or {}).get("name"),
            "gazette": (cert.get("governing_order") or {}).get("notified"),
            "product": None,
        }

    if status == "no_qco_identified":
        return {"scheme": "none", "mandatory": False,
                "explanation": cert.get("message", ""),
                "qco": None, "gazette": None, "product": None}

    # Unknown: never reported as a clearance.
    return {"scheme": "not_verified", "mandatory": False,
            "explanation": cert.get("message", "Certification status has not been checked."),
            "qco": None, "gazette": None, "product": None}


def _to_result(state: Dict[str, Any], hit: Dict[str, Any],
               explain: bool, rank_score: Optional[float] = None) -> Dict[str, Any]:
    s = hit["standard"]
    sid = s["id"]
    sim = float(hit.get("semantic_similarity") or 0.0)
    version = state["versions"].resolve(sid)
    cert = state["certification"].advise(sid)

    from app.core.citation import format_citation

    successor = None
    latest = version.get("latest_active") or []
    if latest and not version.get("is_current"):
        successor = latest[0]["id"]

    warning = None
    if s.get("verified") is False and s.get("provenance") == "published_text_ocr":
        warning = ("Metadata extracted from scanned text and not verified against "
                   "the BIS catalogue.")

    result = {
        "id": sid,
        "number": _citation_number(s),
        "title": s.get("title", ""),
        # Bounded, and only comparable within one response, as their field
        # says. Derived from the fused rank rather than raw similarity: the
        # ranking uses lexical, alias and designation channels too, so a raw
        # cosine would have shown a higher score on a lower-ranked row.
        "final_score": round(rank_score if rank_score is not None
                             else min(max(sim, 0.0), 1.0), 4),
        "stage_scores": {
            "dense": round(sim, 4),
            "bm25": round(float(hit.get("bm25_score") or 0.0), 4),
            "cross_encoder": round(float(hit.get("rerank_score") or 0.0), 4),
            "ltr_or_fallback": round(float(hit.get("fused_score") or 0.0), 5),
        },
        # This engine has no learning-to-rank stage; their own promotion gate
        # rejected theirs as worse than the cross-encoder alone.
        "ranker_used": "fallback",
        "scope": s.get("scope", "") or "",
        "category": s.get("category", "") or s.get("department", "") or "",
        "status": s.get("status", "active"),
        "version": str(s.get("edition_year") or ""),
        "last_amended": (s.get("amendments") or [{}])[-1].get("date", "") if s.get("amendments") else "",
        "superseded_by": successor,
        "certification": _certification_block(cert),
        "data_warning": warning,
        "citation": format_citation(s) if s.get("edition_year") else s.get("number", ""),
        "amendment_count": len(s.get("amendments") or []),
        "explanation": None,
    }

    if explain:
        got = state["explainer"].generate(s, {
            "confidence_band": "high" if sim >= STRONG_SIMILARITY else "medium",
            "semantic_similarity": sim,
        }, cert)
        result["explanation"] = got.get("explanation")

    return result


def _rank_scores(hits: List[Dict[str, Any]]) -> List[float]:
    """Monotonic [0,1] scores preserving the fused order."""
    if not hits:
        return []
    fused = [float(h.get("reranked_score") or h.get("fused_score") or 0.0) for h in hits]
    hi, lo = max(fused), min(fused)
    span = hi - lo
    if span <= 0:
        return [1.0 for _ in fused]
    # Keep the top result's own similarity as its headline score so the
    # confidence gate stays calibrated against a real similarity, then scale
    # the rest beneath it.
    top_sim = min(max(float(hits[0].get("semantic_similarity") or 0.0), 0.0), 1.0)
    return [round(top_sim * (0.35 + 0.65 * (f - lo) / span), 4) for f in fused]


def _confidence(results: List[Dict[str, Any]], top_similarity: float) -> str:
    if not results:
        return "none"
    top = top_similarity
    if top >= STRONG_SIMILARITY:
        return "strong"
    if top <= NONE_SIMILARITY:
        return "none"
    return "uncertain"


def build_router(state: Dict[str, Any], search) -> APIRouter:
    """`state` and `search` are injected so this stays independent of app
    module globals and can be tested directly."""

    @router.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        if not req.query or not req.query.strip():
            raise HTTPException(400, "query must not be empty")

        hits = search(req.query, top_k=min(req.top_k, 50))
        scores = _rank_scores(hits)
        results = [_to_result(state, h, req.explain, scores[i])
                   for i, h in enumerate(hits)]
        top_sim = float(hits[0].get("semantic_similarity") or 0.0) if hits else 0.0
        return {
            "query": req.query,
            "results": results,
            "confidence": _confidence(results, top_sim),
            "translation": {
                "original": req.query,
                "translated_text": req.query,
                "detected_language": req.language or "en",
                "language_name": "English",
                # This engine embeds Indian-language queries directly with a
                # multilingual model instead of translating first, so nothing
                # was translated and saying otherwise would misreport it.
                "translated": False,
                "error": None,
            },
        }

    @router.get("/languages")
    def languages():
        """No translation stage here: the embedding model is multilingual, so
        Indian-language queries are embedded directly. Reported honestly so the
        selector does not imply a capability that is not present."""
        return {
            "languages": [
                {"code": "en", "name": "English", "native_name": "English"},
                {"code": "hi", "name": "Hindi", "native_name": "हिन्दी"},
            ],
            "default": "en",
            "note": ("Queries are embedded directly by a multilingual model; no "
                     "machine translation step is applied."),
        }

    @router.post("/extract")
    async def extract(file: UploadFile = File(...), top_k: int = Query(10)):
        from app.core.documents import UnsupportedDocument, extract_text_with_meta

        data = await file.read()
        try:
            extracted = extract_text_with_meta(
                data, filename=file.filename, content_type=file.content_type)
        except UnsupportedDocument as exc:
            raise HTTPException(400, str(exc))

        text = extracted["text"]
        hits = search(text[:8000], top_k=min(top_k, 50))
        scores = _rank_scores(hits)
        results = [_to_result(state, h, False, scores[i]) for i, h in enumerate(hits)]
        top_sim = float(hits[0].get("semantic_similarity") or 0.0) if hits else 0.0
        return {
            "filename": file.filename,
            "extracted_characters": len(text),
            "extraction_method": extracted["method"],
            "extraction_warnings": extracted["warnings"],
            "query": text[:500],
            "results": results,
            "confidence": _confidence(results, top_sim),
        }

    # /standards is served by the native paginated endpoint in main.py, which
    # accepts both `department` and `category` so one implementation satisfies
    # both this contract and the engine's own.

    def _resolve(id_or_number: str) -> Dict[str, Any]:
        """Resolve an internal id, a bare number, or a full citation.

        The UI links standards as "IS 456:2000" -- number and edition together,
        which is how a citation is written. Records store those separately, so
        matching only on `number` 404'd every detail page reached from a
        citation.
        """
        by_id = state["allied"].by_id
        if id_or_number in by_id:
            return by_id[id_or_number]

        wanted = id_or_number.strip()
        year = None
        if ":" in wanted:
            head, _, tail = wanted.rpartition(":")
            tail = tail.strip()
            if tail.isdigit() and len(tail) == 4:
                wanted, year = head.strip(), int(tail)

        lowered = wanted.lower()
        matches = [s for s in state["standards"]
                   if s.get("number", "").strip().lower() == lowered]
        if year is not None:
            exact = [s for s in matches if s.get("edition_year") == year]
            if exact:
                return exact[0]
        if matches:
            # Prefer the citable edition when the citation named one we lack.
            active = [s for s in matches if s.get("status") == "active"]
            pool = active or matches
            return max(pool, key=lambda s: s.get("edition_year") or 0)

        raise HTTPException(404, f"unknown standard {id_or_number}")

    @router.get("/standards/{id_or_number}")
    def standard_detail(id_or_number: str):
        s = _resolve(id_or_number)
        cert = state["certification"].advise(s["id"])
        return {
            **{k: s.get(k) for k in ("id", "title", "scope", "status",
                                     "committee", "department", "provenance",
                                     "verified", "source_url")},
            "number": _citation_number(s),
            "version": str(s.get("edition_year") or ""),
            "certification": _certification_block(cert),
            "version_info": state["versions"].resolve(s["id"]),
        }

    @router.get("/standards/{id_or_number}/related")
    def standard_related(id_or_number: str):
        """Allied cluster in the shape the detail page renders.

        Their contract groups forward dependencies by relation type, separates
        inverse references, and carries a `researched` flag plus per-item
        `outside_corpus`. That is a better shape than a flat list: it can say
        "this standard cites IS 383, which we do not hold" rather than quietly
        omitting it -- and 3,009 extracted references point outside this
        corpus.
        """
        s_rec = _resolve(id_or_number)
        grouped = state["allied"].grouped_allied(s_rec["id"])

        depends_on = []
        for rel, edges in grouped.items():
            forward = [e for e in edges if e.get("direction") == "forward"]
            if not forward:
                continue
            depends_on.append({
                "type": rel,
                "heading": RELATION_HEADINGS.get(rel, rel.replace("_", " ").title()),
                "explanation": RELATION_EXPLANATIONS.get(rel),
                "standards": [{
                    "id": e["id"], "number": e["number"], "title": e["title"],
                    "status": e.get("status", "active"),
                    "note": e.get("relation_note", ""),
                    "outside_corpus": False,
                } for e in forward],
            })

        referenced_by = [{
            "id": e["id"], "number": e["number"], "title": e["title"],
            "status": e.get("status", "active"),
            "note": e.get("relation_note", ""),
        } for edges in grouped.values() for e in edges
            if e.get("direction") == "inverse"]

        # References read from the document's annex that name a standard this
        # corpus does not hold. Shown rather than dropped.
        held = {e["number"] for g in depends_on for e in g["standards"]}
        outside = []
        for ref in (s_rec.get("referred_standards") or []):
            num = f"IS {ref['number']}" + (f" (Part {ref['part']})" if ref.get("part") else "")
            if num not in held:
                outside.append({"id": None, "number": num, "title": "",
                                "status": "unknown", "note": "",
                                "outside_corpus": True})
        if outside:
            depends_on.append({
                "type": "outside_corpus",
                "heading": "Cited but not in this corpus",
                "explanation": ("Named in this standard's referred-standards annex. "
                                "This corpus does not hold them, so nothing is claimed "
                                "about their content or status."),
                "standards": outside,
            })

        total = sum(len(g["standards"]) for g in depends_on) + len(referenced_by)
        return {
            "id": s_rec["id"],
            "number": _citation_number(s_rec),
            "total": total,
            # Absence of edges is not a claim that a standard stands alone.
            "researched": bool(s_rec.get("allied")) or bool(s_rec.get("referred_standards")),
            "depends_on": depends_on,
            "referenced_by": referenced_by,
        }

    @router.get("/standards/{id_or_number}/amendments")
    def standard_amendments(id_or_number: str):
        s = _resolve(id_or_number)
        amendments = s.get("amendments") or []
        return {
            "id": s["id"],
            "number": _citation_number(s),
            "amendments": amendments,
            "count": len(amendments),
            # Absence of amendment records is not a statement that none exist.
            "checked": bool(s.get("qco_checked")) or bool(amendments),
        }

    return router
