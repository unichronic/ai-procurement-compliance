from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv

from app.core.retrieval import StandardsIndex
from app.core.allied import AlliedGraph
from app.core.versioning import VersionResolver
from app.core.certification import CertificationAdvisor
from app.core.batch import split_document
from app.core.explain import ExplanationGenerator
from app.core.lint import SpecLinter

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

_state: Dict[str, Any] = {}


def _load_data():
    standards = json.loads((DATA_DIR / "standards.json").read_text())["standards"]
    qco_orders = json.loads((DATA_DIR / "qco_orders.json").read_text())["orders"]
    return standards, qco_orders


@asynccontextmanager
async def lifespan(app: FastAPI):
    standards, qco_orders = _load_data()

    index = StandardsIndex(standards)
    index.build()

    _state["standards"] = standards
    _state["qco_orders"] = qco_orders
    _state["index"] = index
    _state["allied"] = AlliedGraph(standards)
    _state["versions"] = VersionResolver(standards)
    _state["certification"] = CertificationAdvisor(standards, qco_orders)
    _state["explainer"] = ExplanationGenerator()
    _state["linter"] = SpecLinter(
        standards=standards,
        index=index,
        versions=_state["versions"],
        allied=_state["allied"],
        certification=_state["certification"],
    )
    yield
    _state.clear()


app = FastAPI(
    title="SIH26108 — Indian Standards Recommendation Engine",
    description=(
        "AI-Powered Recommendation Engine for Identifying Applicable Indian "
        "Standards for Procurement Specifications. Prototype for SIH 2026 "
        "Problem Statement 108 (Dept. of Consumer Affairs / BIS)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _confidence_band(semantic_similarity: float) -> str:
    if semantic_similarity >= 0.55:
        return "high"
    if semantic_similarity >= 0.35:
        return "medium"
    return "low"


def _build_recommendation(hit: Dict[str, Any]) -> Dict[str, Any]:
    s = hit["standard"]
    sid = s["id"]
    version_info = _state["versions"].resolve(sid)
    cert_info = _state["certification"].advise(sid)
    allied = _state["allied"].grouped_allied(sid)

    return {
        "id": sid,
        "number": s["number"],
        "edition_year": s["edition_year"],
        "title": s["title"],
        "scope": s["scope"],
        "committee": s["committee"],
        "department": s["department"],
        "data_confidence": s.get("data_confidence", "unknown"),
        "match": {
            "confidence_band": _confidence_band(hit["semantic_similarity"]),
            "semantic_similarity": round(hit["semantic_similarity"], 4),
            "lexical_rank": hit["lexical_rank"],
            "semantic_rank": hit["semantic_rank"],
            "fused_score": round(hit["fused_score"], 5),
        },
        "version": version_info,
        "certification": cert_info,
        "allied_standards": allied,
    }


class RecommendRequest(BaseModel):
    text: str
    top_k: int = 5


class BatchRequest(BaseModel):
    document_text: str
    top_k_per_item: int = 3


class LintRequest(BaseModel):
    document_text: str
    suggest_missing: bool = True


class ExplainRequest(BaseModel):
    standard_id: str
    confidence_band: Optional[str] = None
    semantic_similarity: Optional[float] = None
    lexical_rank: Optional[int] = None
    semantic_rank: Optional[int] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "standards_loaded": len(_state["standards"]),
        "qco_orders_loaded": len(_state["qco_orders"]),
        "embedding_model": _state["index"].model_name,
        "explanation_api": "groq" if _state["explainer"].available else "template (no GROQ_API_KEY set)",
        "note": (
            "Embedding + retrieval run locally; no procurement/tender text is sent to any "
            "external API. The explanation layer, when enabled, only ever receives "
            "already-matched public standard metadata and numeric match scores."
        ),
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "text must not be empty")

    hits = _state["index"].search(req.text, top_k=req.top_k)
    recommendations = [_build_recommendation(h) for h in hits]

    return {
        "query": req.text,
        "recommendations": recommendations,
    }


@app.post("/batch")
def batch_recommend(req: BatchRequest):
    items = split_document(req.document_text)
    if not items:
        raise HTTPException(400, "Could not extract any line items from document_text")

    results = []
    for idx, line in enumerate(items):
        hits = _state["index"].search(line, top_k=req.top_k_per_item)
        results.append({
            "item_index": idx,
            "line": line,
            "recommendations": [_build_recommendation(h) for h in hits],
        })

    return {
        "item_count": len(results),
        "items": results,
    }


@app.post("/lint")
def lint_specification(req: LintRequest):
    """Audit a draft tender spec and report defects.

    The inverse of /recommend: instead of asking what applies, this checks what
    the draft already says and reports superseded citations, omitted normative
    references, missing mandatory conformity marks, and items that cite no
    standard at all.
    """
    if not req.document_text or not req.document_text.strip():
        raise HTTPException(400, "document_text must not be empty")

    result = _state["linter"].lint(req.document_text, suggest_missing=req.suggest_missing)
    return result


@app.post("/explain")
def explain(req: ExplainRequest):
    """Generate a plain-language explanation for an already-computed match.

    Takes only the standard_id plus the numeric match signals the client
    already received from /recommend or /batch -- never raw procurement
    text -- and returns a human-readable explanation grounded in that
    standard's public metadata.
    """
    standard = _state["allied"].by_id.get(req.standard_id)
    if not standard:
        raise HTTPException(404, f"unknown standard id {req.standard_id}")

    match = {
        "confidence_band": req.confidence_band,
        "semantic_similarity": req.semantic_similarity,
        "lexical_rank": req.lexical_rank,
        "semantic_rank": req.semantic_rank,
    }
    certification = _state["certification"].advise(req.standard_id)

    result = _state["explainer"].generate(standard, match, certification)
    return {"standard_id": req.standard_id, **result}


@app.get("/standard/{standard_id}")
def get_standard(standard_id: str):
    s = _state["allied"].by_id.get(standard_id)
    if not s:
        raise HTTPException(404, f"unknown standard id {standard_id}")

    return {
        **s,
        "version": _state["versions"].resolve(standard_id),
        "certification": _state["certification"].advise(standard_id),
        "allied_standards": _state["allied"].grouped_allied(standard_id),
    }


@app.get("/standards")
def list_standards(department: Optional[str] = None):
    standards = _state["standards"]
    if department:
        standards = [s for s in standards if s["department"] == department]
    return {
        "count": len(standards),
        "standards": [
            {"id": s["id"], "number": s["number"], "title": s["title"],
             "department": s["department"], "status": s["status"]}
            for s in standards
        ],
    }


@app.get("/qco-orders")
def list_qco_orders():
    return {"count": len(_state["qco_orders"]), "orders": _state["qco_orders"]}


# Serves the frontend at /ui/ (same origin as the API, so no CORS needed for
# the UI itself). Mounted last so it never shadows an API route above.
if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
