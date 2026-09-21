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

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# Roughly 2x the best possible RRF contribution (2/61), so a single designation
# hit reliably outranks a standard that merely scored well on both soft channels.
DESIGNATION_WEIGHT = 0.05

# Superseded and withdrawn editions are demoted, not removed: they are still
# worth surfacing (the linter has to recognise one when a draft cites it), just
# never ahead of an edition that can actually be cited.
SUPERSEDED_PENALTY = float(os.environ.get("SUPERSEDED_PENALTY", "0.6"))

# Cached document embeddings, keyed by a hash of the exact text encoded.
EMBEDDING_CACHE_DIR = Path(
    os.environ.get("EMBEDDING_CACHE_DIR", Path(__file__).resolve().parents[1] / ".cache" / "embeddings")
)
USE_EMBEDDING_CACHE = os.environ.get("USE_EMBEDDING_CACHE", "1") != "0"

# Max-pooling over chunks gives chunk-rich standards more chances at a high
# score regardless of relevance, and only part of the corpus has ingested full
# text. Damping makes a chunk match win only when it is clearly better than the
# standard's own metadata match, which removes that bias. Tuned on the eval set
# (see README) -- undamped max-pool measurably regressed overall R@1.
CHUNK_SCORE_DAMPING = float(os.environ.get("CHUNK_SCORE_DAMPING", "0.88"))


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
        self._chunk_embeddings: np.ndarray | None = None
        self._chunk_owner: List[int] = []
        self.cache_hit = False

    def _doc_text(self, s: Dict[str, Any]) -> str:
        # Aliases go into the indexed text because the hard case in procurement
        # is market vocabulary that never appears in a BIS title ("MS plate",
        # "hard hat", "peene ka paani"). Both channels need to see them.
        base = f"{s['number']} {s['title']}. {s['scope']}"
        aliases = s.get("aliases") or []
        if aliases:
            base = f"{base} Also known as: {', '.join(aliases)}."
        return base

    def _cache_key(self, docs: List[str], chunk_texts: List[str]) -> str:
        """Content hash of everything the embeddings depend on.

        Keyed on the model name and the exact text encoded, so a corpus edit,
        an alias change or a model swap all invalidate it automatically. There
        is no manual "remember to clear the cache" step to forget.
        """
        h = hashlib.sha256()
        h.update(self._model_name.encode())
        for text in docs:
            h.update(b"\x00")
            h.update(text.encode("utf-8"))
        for text in chunk_texts:
            h.update(b"\x01")
            h.update(text.encode("utf-8"))
        return h.hexdigest()[:24]

    def _load_cache(self, key: str):
        path = EMBEDDING_CACHE_DIR / f"{key}.npz"
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                chunks = data["chunks"] if "chunks" in data.files else None
                if chunks is not None and chunks.size == 0:
                    chunks = None
                return data["docs"], chunks
        except Exception as exc:
            logger.warning("embedding cache unreadable (%s); rebuilding", exc)
            return None  # a corrupt cache must never break startup

    def _save_cache(self, key: str, doc_emb, chunk_emb) -> None:
        try:
            EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = EMBEDDING_CACHE_DIR / f"{key}.npz"
            # Write via a temp file so a crash mid-write can't leave a
            # half-written cache that later loads as garbage. The temp name
            # must itself end in .npz: np.savez appends the extension when it
            # is missing, so a ".npz.tmp" target silently becomes
            # ".npz.tmp.npz" and the rename below then finds nothing.
            tmp = EMBEDDING_CACHE_DIR / f"{key}.tmp.npz"
            np.savez(
                tmp,
                docs=doc_emb,
                chunks=chunk_emb if chunk_emb is not None else np.empty((0, 0)),
            )
            tmp.replace(path)
        except Exception as exc:
            # Caching is an optimisation and must never break startup, but a
            # silent failure means a permanent slow boot nobody diagnoses.
            logger.warning("embedding cache write failed (%s); continuing uncached", exc)

    def build(self) -> None:
        docs = [self._doc_text(s) for s in self.standards]

        # Chunks carry the document's own requirement clauses. Lexically they
        # belong in the same BM25 document (more vocabulary, same standard);
        # semantically they must stay separate, because averaging a whole
        # standard into one vector washes out the specific passage that
        # actually answers the query.
        tokenized = []
        for idx, doc in enumerate(docs):
            chunk_text = " ".join(
                c.get("text", "") for c in (self.standards[idx].get("chunks") or [])
            )
            tokenized.append(_tokenize(f"{doc} {chunk_text}"))
        self._bm25 = BM25Okapi(tokenized)

        self._designations = [
            [d.lower() for d in (s.get("designations") or [])] for s in self.standards
        ]

        chunk_texts: List[str] = []
        self._chunk_owner = []
        for idx, s in enumerate(self.standards):
            for chunk in (s.get("chunks") or []):
                text = chunk.get("text")
                if text:
                    chunk_texts.append(text)
                    self._chunk_owner.append(idx)

        # Encoding 6,383 documents takes ~50s on CPU, on every boot and every
        # test session. The corpus changes far less often than the process
        # restarts, so the embeddings are cached against a hash of their input.
        cache_key = self._cache_key(docs, chunk_texts)
        cached = self._load_cache(cache_key) if USE_EMBEDDING_CACHE else None

        if cached is not None:
            self._embeddings, self._chunk_embeddings = cached
            self._model = None  # loaded lazily; queries still need to encode
            self.cache_hit = True
            return

        self._model = SentenceTransformer(self._model_name)
        self._embeddings = self._model.encode(
            docs, normalize_embeddings=True, show_progress_bar=False
        )

        self._chunk_embeddings = (
            self._model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
            if chunk_texts else None
        )

        if USE_EMBEDDING_CACHE:
            self._save_cache(cache_key, self._embeddings, self._chunk_embeddings)

    def _ensure_query_model(self) -> SentenceTransformer:
        """The query encoder, loaded on demand.

        A cache hit skips building document embeddings but queries still have
        to be encoded, so the model loads on first search rather than at
        startup — which is what keeps a warm boot fast.
        """
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _semantic_scores(self, q_emb: np.ndarray) -> np.ndarray:
        """Metadata score, max-pooled against the standard's own chunks.

        Max rather than mean: the question is "does this standard contain a
        passage that answers the query", and one strongly matching clause is
        the answer even when the other seven clauses are irrelevant.
        """
        scores = self._embeddings @ q_emb
        if self._chunk_embeddings is None:
            return scores

        chunk_scores = self._chunk_embeddings @ q_emb * CHUNK_SCORE_DAMPING
        pooled = scores.copy()
        for chunk_idx, owner in enumerate(self._chunk_owner):
            if chunk_scores[chunk_idx] > pooled[owner]:
                pooled[owner] = chunk_scores[chunk_idx]
        return pooled

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
        assert self._bm25 is not None, "Index not built"

        # Lexical ranking
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        has_lexical_signal = bool(bm25_scores.max() > 0)
        lexical_order = np.argsort(-bm25_scores)

        # Semantic ranking
        q_emb = self._ensure_query_model().encode(
            [query], normalize_embeddings=True, show_progress_bar=False)[0]
        sem_scores = self._semantic_scores(q_emb)
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

        # Demote editions that cannot be cited. Superseded and withdrawn
        # editions share a number, title and most of their scope with the
        # current one, so they score almost identically -- and returning the
        # outdated edition first is the exact failure (citing a superseded
        # standard) this project exists to stop.
        #
        # This was originally a tie-break, which only fires on an exact score
        # match. That held at 39 standards, where the two editions scored
        # identically, and silently stopped working at 6,383, where slightly
        # different lexical ranks let the superseded edition win outright. A
        # penalty is the property actually wanted; a tie-break only approximated
        # it on a small corpus.
        for idx in range(len(self.standards)):
            status = self.standards[idx].get("status", "active")
            if status != "active":
                rrf[idx] *= SUPERSEDED_PENALTY

        ranked = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]

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
