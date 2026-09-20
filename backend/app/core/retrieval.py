"""
Hybrid lexical + semantic retrieval over the standards metadata.

Deliberately uses a LOCAL, open-weight multilingual embedding model
(paraphrase-multilingual-MiniLM-L12-v2) instead of an external LLM API
(OpenAI/Groq/etc). Two reasons:
  1. Data sovereignty -- tender/procurement text can be sensitive
     pre-publication; nothing here leaves the process boundary.
  2. It gives free multilingual support (Hindi and other Indian-language
     queries embed into the same space as English) without a separate
     translation step.

Combines BM25 (catches exact IS-number / known-term hits) with semantic
cosine similarity (catches paraphrased, no-vocabulary-overlap queries)
via Reciprocal Rank Fusion, so neither signal alone can starve the other.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# Roughly 2x the best possible RRF contribution (2/61), so a single designation
# hit reliably outranks a standard that merely scored well on both soft channels.
DESIGNATION_WEIGHT = 0.05


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class StandardsIndex:
    def __init__(self, standards: List[Dict[str, Any]], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.standards = standards
        self.by_id = {s["id"]: s for s in standards}
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None
        self._designations: List[List[str]] = []

    def _doc_text(self, s: Dict[str, Any]) -> str:
        # Aliases go into the indexed text because the hard case in procurement
        # is market vocabulary that never appears in a BIS title ("MS plate",
        # "hard hat", "peene ka paani"). Both channels need to see them.
        base = f"{s['number']} {s['title']}. {s['scope']}"
        aliases = s.get("aliases") or []
        if aliases:
            base = f"{base} Also known as: {', '.join(aliases)}."
        return base

    def build(self) -> None:
        docs = [self._doc_text(s) for s in self.standards]

        tokenized = [_tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized)

        self._designations = [
            [d.lower() for d in (s.get("designations") or [])] for s in self.standards
        ]

        self._model = SentenceTransformer(self._model_name)
        self._embeddings = self._model.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        )

    def _designation_scores(self, query: str) -> np.ndarray:
        """Symbolic channel: exact hits on technical designations (E250, IE3,
        M25, 22K, 14.2 kg). These are near-unique identifiers rather than fuzzy
        evidence -- when one fires it should be able to outweigh a soft semantic
        score, so it's fused separately instead of being left to BM25 (which
        treats 'e250' as just another token with low IDF weight)."""
        q = query.lower()
        scores = np.zeros(len(self.standards), dtype=float)
        for idx, designations in enumerate(self._designations):
            scores[idx] = sum(1.0 for d in designations if d and d in q)
        return scores

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def size(self) -> int:
        return len(self.standards)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        assert self._bm25 is not None and self._model is not None, "Index not built"

        # Lexical ranking
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        has_lexical_signal = bool(bm25_scores.max() > 0)
        lexical_order = np.argsort(-bm25_scores)

        # Semantic ranking
        q_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sem_scores = self._embeddings @ q_emb
        semantic_order = np.argsort(-sem_scores)

        # Reciprocal Rank Fusion (k=60 is the standard RRF damping constant).
        # When the query has no lexical signal at all -- e.g. a non-Latin-script
        # query the BM25 tokenizer regex can't match (Hindi/other Indian-language
        # input), or vocabulary with zero term overlap -- argsort on an all-zero
        # score array still returns *some* order, an artifact of tie-breaking
        # (effectively document array order), not real evidence. Folding that
        # into RRF would let array position masquerade as a match signal and
        # can outrank a correct top semantic hit, so in that case rank purely
        # on the semantic score instead.
        k = 60
        rrf: Dict[int, float] = {}
        lexical_rank = {idx: rank for rank, idx in enumerate(lexical_order)}
        semantic_rank = {idx: rank for rank, idx in enumerate(semantic_order)}
        for idx in range(len(self.standards)):
            sr = semantic_rank[idx]
            if has_lexical_signal:
                rrf[idx] = 1.0 / (k + lexical_rank[idx] + 1) + 1.0 / (k + sr + 1)
            else:
                rrf[idx] = 1.0 / (k + sr + 1)

        # Symbolic channel, added on top rather than rank-fused: a designation
        # hit is hard evidence, so it should lift a standard past soft scores
        # instead of contributing a fractional rank term.
        designation_scores = self._designation_scores(query)
        for idx in range(len(self.standards)):
            if designation_scores[idx]:
                rrf[idx] += DESIGNATION_WEIGHT * designation_scores[idx]

        # Tie-break toward editions that can actually be cited. Superseded and
        # withdrawn editions share a number, title and most of their scope with
        # the current one, so they score almost identically -- without this the
        # engine happily returns the outdated edition first, which is the exact
        # failure mode (citing superseded standards) this project exists to stop.
        def sort_key(item):
            idx, score = item
            status = self.standards[idx].get("status", "active")
            return (-score, 0 if status == "active" else 1)

        ranked = sorted(rrf.items(), key=sort_key)[:top_k]

        results = []
        for idx, fused_score in ranked:
            s = self.standards[idx]
            results.append({
                "standard": s,
                "fused_score": float(fused_score),
                "lexical_rank": (int(lexical_rank[idx]) + 1) if has_lexical_signal else None,
                "semantic_rank": int(semantic_rank[idx]) + 1,
                "semantic_similarity": float(sem_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "designation_hits": int(designation_scores[idx]),
            })
        return results
