"""
LLM-based plain-language explanation layer, backed by the Groq API.

Deliberately receives ONLY already-matched, already-public standard
metadata and numeric match signals (similarity score, ranks, certification
flags) -- never the raw procurement/tender text that produced the match.
The retrieval that actually touches that sensitive text stays entirely
local (see retrieval.py, a self-hosted multilingual embedding model);
this module's only job is to narrate an already-computed result in plain
language, so nothing sensitive ever crosses the process boundary to Groq.

Falls back to a deterministic template if no GROQ_API_KEY is configured,
the `groq` package isn't installed, or the API call fails -- the app must
degrade gracefully rather than break a demo on a missing key or a flaky
network.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

_SYSTEM_PROMPT = (
    "You explain, in plain language for a government procurement officer, "
    "why a specific Indian Standard (IS) was recommended by a search system. "
    "You are given only the standard's public metadata and numeric "
    "match-confidence signals -- you have NOT seen and must NOT speculate "
    "about the underlying procurement text that produced the match. "
    "In 2-3 sentences: (1) state what the standard covers, (2) explain what "
    "the confidence band/score implies about match strength, (3) flag "
    "anything procurement-relevant (mandatory certification, superseded "
    "version) present in the data. Do not invent facts not present in the "
    "provided fields. Respond in plain prose only -- no markdown, no "
    "asterisks, no bullet points, no headings."
)


class ExplanationGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b"):
        self._api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        self._model = model
        self._client = None
        if self._api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(
        self,
        standard: Dict[str, Any],
        match: Dict[str, Any],
        certification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        template = self._template(standard, match, certification)
        if self._client is None:
            return {"explanation": template, "source": "template"}

        fields = {
            "standard_number": standard.get("number"),
            "title": standard.get("title"),
            "scope": standard.get("scope"),
            "committee": standard.get("committee"),
            "department": standard.get("department"),
            "status": standard.get("status"),
            "confidence_band": match.get("confidence_band"),
            "semantic_similarity": match.get("semantic_similarity"),
            "lexical_rank": match.get("lexical_rank"),
            "semantic_rank": match.get("semantic_rank"),
            "mandatory_certification": (certification or {}).get("mandatory_certification"),
            "certification_scheme": (certification or {}).get("scheme"),
        }

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": repr(fields)},
                ],
                temperature=0.2,
                max_tokens=300,
                reasoning_effort="low",
            )
            text = resp.choices[0].message.content.strip()
            if not text:
                return {"explanation": template, "source": "template"}
            return {"explanation": text, "source": "groq", "model": self._model}
        except Exception as exc:
            return {"explanation": template, "source": "template", "fallback_reason": str(exc)}

    def _template(
        self,
        standard: Dict[str, Any],
        match: Dict[str, Any],
        certification: Optional[Dict[str, Any]] = None,
    ) -> str:
        band = match.get("confidence_band", "unknown")
        sim = match.get("semantic_similarity")
        sim_str = f"{sim:.2f}" if isinstance(sim, (int, float)) else "n/a"
        parts = [
            f"{standard.get('number')} covers: {standard.get('scope')}",
            f"Match confidence is {band} (semantic similarity {sim_str}).",
        ]
        if certification and certification.get("mandatory_certification"):
            parts.append(
                f"Note: {certification.get('scheme')} certification is mandatory for this category."
            )
        return " ".join(parts)
