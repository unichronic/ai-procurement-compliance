"""Cross-encoder reranking of the candidate set.

RRF fusion scores a query and a standard independently and then combines ranks.
A cross-encoder instead reads the pair jointly, which is slower but much better
at deciding which of several plausible candidates actually answers the query.

That is precisely this system's measured weakness: on the held-out benchmark
the correct standard is in the top 5 about 95% of the time but ranked first
only 64% of the time. Reranking can only move candidates the first stage
already retrieved, so it targets exactly that gap and nothing else.

Approach adapted from the team's `standards-retrieval/retrieval/rerank.py`;
reimplemented here against this codebase's data model rather than imported.

`ms-marco-MiniLM` is English-only, which matters because multilingual input is
a first-class requirement. Two defences: Devanagari queries skip reranking
entirely, and the remainder fuse the cross-encoder's ranking with the first
stage's rather than replacing it.

Measured on the held-out set (22 queries, `--heldout --rerank`):

    baseline            R@1 0.64  R@5 0.95  MRR 0.759
    rerank, overriding  R@1 0.77  R@5 0.91  MRR 0.841   <- loses recall
    rerank, fused k=5   R@1 0.73  R@5 0.95  MRR 0.841   <- shipped

Overriding scored higher on R@1 but dropped a correct answer out of the top 5
and cut code-mixed MRR from 0.600 to 0.500. Fusing keeps the gain, restores
code-mixed to 1.000, and costs no recall. The in-lexicon set is unchanged
either way.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from app.core.retrieval import SUPERSEDED_PENALTY

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Rerank this many RRF candidates. Beyond the first stage's reach it can't help,
# and cross-encoder cost is linear in candidates. Measured on the held-out set:
#
#   depth 20   R@1 0.73  R@5 0.95  MRR 0.841
#   depth 10   R@1 0.73  R@5 1.00  MRR 0.839   <- shipped, ~1.8x faster
#   depth  5   R@1 0.68  R@5 0.91  MRR 0.799
#
# Depth 10 gives up 0.002 MRR to eliminate the last complete miss. A deeper
# pool hands the cross-encoder more weak candidates to mistakenly promote.
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "10"))

# Damping for fusing the cross-encoder's ranking with the first stage's. Lower
# values let the reranker move candidates further; tuned on the held-out set.
RERANK_RRF_K = int(os.environ.get("RERANK_RRF_K", "5"))

_LATIN_RE = re.compile(r"[A-Za-z]")


def is_latin_script(text: str) -> bool:
    """Whether an English-only reranker can meaningfully read this query."""
    letters = _LATIN_RE.findall(text)
    return len(letters) >= max(3, 0.5 * len(re.findall(r"\w", text)))


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, enabled: bool = True):
        self.model_name = model_name
        self._enabled = enabled
        self._model = None
        self._load_failed = False

    @property
    def available(self) -> bool:
        if not self._enabled or self._load_failed:
            return False
        return self._ensure_model() is not None

    def _ensure_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except Exception:
            # No network, no cached weights, or an incompatible install: the
            # engine must still answer, just without reranking.
            self._load_failed = True
            self._model = None
        return self._model

    def _pair_text(self, standard: Dict[str, Any]) -> str:
        parts = [standard.get("number", ""), standard.get("title", "")]
        scope = standard.get("scope")
        if scope:
            parts.append(scope)
        aliases = standard.get("aliases") or []
        if aliases:
            parts.append("Also known as: " + ", ".join(aliases[:8]))
        return ". ".join(p for p in parts if p)

    def rerank(self, query: str, hits: List[Dict[str, Any]],
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Reorder `hits` in place of the first-stage order.

        Returns the input untouched when reranking can't or shouldn't run, so
        callers never need to branch on availability.
        """
        # Every early return still honours top_k: callers retrieve deeper than
        # they want so the reranker has candidates to work with, so returning
        # the untruncated list would hand back the whole candidate pool.
        def passthrough():
            return hits[:top_k] if top_k else hits

        if not hits or not query.strip():
            return passthrough()
        if not is_latin_script(query):
            return passthrough()

        model = self._ensure_model()
        if model is None:
            return passthrough()

        candidates = hits[:RERANK_CANDIDATES]
        pairs = [(query, self._pair_text(h["standard"])) for h in candidates]

        try:
            scores = model.predict(pairs)
        except Exception:
            return passthrough()

        for hit, score in zip(candidates, scores):
            hit["rerank_score"] = float(score)

        # Fuse rather than override. Letting the cross-encoder dictate the order
        # outright measurably cost recall@5 and hurt code-mixed queries, because
        # an English-only model confidently discards candidates it cannot read
        # (romanised Hindi arrives in Latin script, so it passes the script
        # check above). Treating it as one more rank channel in the same RRF
        # scheme the retriever already uses keeps its gain on English trade
        # names without letting it throw away first-stage evidence.
        rerank_order = sorted(range(len(candidates)),
                              key=lambda i: -candidates[i]["rerank_score"])
        rerank_rank = {idx: rank for rank, idx in enumerate(rerank_order)}

        k = RERANK_RRF_K
        for idx, hit in enumerate(candidates):
            hit["rerank_rank"] = rerank_rank[idx] + 1
            score = 1.0 / (k + idx + 1) + 1.0 / (k + rerank_rank[idx] + 1)
            # The cross-encoder has no notion of whether an edition is current,
            # and will happily promote a superseded one back above its own
            # successor -- they read almost identically. Re-applying the
            # retriever's demotion here keeps that guarantee end to end.
            if hit["standard"].get("status", "active") != "active":
                score *= SUPERSEDED_PENALTY
            hit["reranked_score"] = score

        reranked = sorted(candidates, key=lambda h: -h["reranked_score"])
        out = reranked + hits[RERANK_CANDIDATES:]
        return out[:top_k] if top_k else out
