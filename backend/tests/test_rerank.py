import pytest

from app.core.rerank import CrossEncoderReranker, is_latin_script


class _FakeModel:
    """Scores by position in a caller-supplied preference list, so tests assert
    fusion behaviour rather than a real model's opinions."""

    def __init__(self, preferred_order):
        self.preferred_order = preferred_order
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        scores = []
        for _, doc in pairs:
            rank = next(
                (i for i, token in enumerate(self.preferred_order) if token in doc),
                len(self.preferred_order),
            )
            scores.append(1.0 - rank * 0.1)
        return scores


def _hits(*numbers):
    return [
        {"standard": {"id": f"IS_{n}", "number": f"IS {n}", "title": f"Title {n}",
                      "scope": f"Scope for {n}"},
         "semantic_similarity": 0.5, "fused_score": 0.03,
         "lexical_rank": i + 1, "semantic_rank": i + 1,
         "bm25_score": 1.0, "designation_hits": 0}
        for i, n in enumerate(numbers)
    ]


def _reranker(preferred_order):
    r = CrossEncoderReranker()
    r._model = _FakeModel(preferred_order)
    return r


@pytest.mark.parametrize("text,expected", [
    ("XLPE cable for substation", True),
    ("helmet ka ISI standard kya hai", True),      # romanised Hindi is Latin
    ("पीने के पानी के लिए मानक", False),
    ("सोने के आभूषण", False),
])
def test_script_detection(text, expected):
    assert is_latin_script(text) is expected


def test_devanagari_query_skips_reranking():
    """An English-only cross-encoder must not be allowed to score a query it
    cannot read."""
    r = _reranker(["IS 9999"])
    hits = _hits("1111", "2222")
    out = r.rerank("पीने के पानी के लिए मानक", hits, top_k=2)

    assert [h["standard"]["id"] for h in out] == ["IS_1111", "IS_2222"]
    assert r._model.calls == []


def test_reranking_promotes_preferred_candidate():
    """Fusion is symmetric, so the reranker overtakes the first stage's top hit
    only when it prefers a candidate by a clear margin -- here, last to first."""
    r = _reranker(["IS 5555"])
    hits = _hits("1111", "2222", "3333", "4444", "5555")
    out = r.rerank("some english query", hits, top_k=5)

    ids = [h["standard"]["id"] for h in out]
    assert ids.index("IS_5555") < ids.index("IS_3333"), "preferred candidate did not move up"
    assert out[0]["standard"]["id"] in ("IS_5555", "IS_1111")


def test_passthrough_paths_respect_top_k():
    """Regression: callers over-retrieve so the reranker has candidates, so an
    early return that skipped truncation handed back the whole pool (a Hindi
    query asking for 3 results got 20)."""
    r = _reranker(["IS 1111"])
    hits = _hits(*[str(i) * 4 for i in range(1, 9)])

    assert len(r.rerank("पीने के पानी के लिए मानक", hits, top_k=3)) == 3

    disabled = CrossEncoderReranker(enabled=False)
    assert len(disabled.rerank("english query", hits, top_k=3)) == 3


def test_fusion_preserves_first_stage_evidence():
    """Fused, not overriding: a candidate the reranker dislikes but that the
    first stage ranked top must not be discarded outright. Overriding measurably
    cost recall@5 on the held-out set."""
    r = _reranker(["IS_NOT_PRESENT"])  # reranker is indifferent to everything
    hits = _hits("1111", "2222", "3333")
    out = r.rerank("some english query", hits, top_k=3)
    assert out[0]["standard"]["id"] == "IS_1111"


def test_rerank_annotates_hits_with_scores():
    r = _reranker(["IS 2222"])
    out = r.rerank("english query", _hits("1111", "2222"), top_k=2)
    assert all("rerank_score" in h and "rerank_rank" in h for h in out)


def test_respects_top_k():
    r = _reranker(["IS 3333"])
    out = r.rerank("english query", _hits("1111", "2222", "3333"), top_k=2)
    assert len(out) == 2


def test_model_failure_falls_back_to_first_stage_order():
    """A reranker failure must degrade to unranked results, never to an error."""
    class Boom:
        def predict(self, pairs):
            raise RuntimeError("model exploded")

    r = CrossEncoderReranker()
    r._model = Boom()
    hits = _hits("1111", "2222")
    out = r.rerank("english query", hits, top_k=2)
    assert [h["standard"]["id"] for h in out] == ["IS_1111", "IS_2222"]


def test_unavailable_model_reports_unavailable_and_passes_through():
    r = CrossEncoderReranker(enabled=False)
    assert r.available is False
    hits = _hits("1111", "2222")
    assert r.rerank("english query", hits, top_k=2) == hits


def test_empty_inputs_are_safe():
    r = _reranker(["IS 1111"])
    assert r.rerank("", _hits("1111"), top_k=1) == _hits("1111")
    assert r.rerank("query", [], top_k=1) == []


def test_recommend_hindi_query_returns_requested_count(client):
    resp = client.post("/recommend",
                       json={"text": "निर्माण में उपयोग होने वाले स्ट्रक्चरल स्टील के लिए मानक",
                             "top_k": 3})
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) == 3


def test_recommend_endpoint_accepts_rerank_flag(client):
    off = client.post("/recommend", json={"text": "structural steel", "top_k": 3,
                                          "rerank": False})
    on = client.post("/recommend", json={"text": "structural steel", "top_k": 3,
                                         "rerank": True})
    assert off.status_code == 200 and on.status_code == 200
    assert len(off.json()["recommendations"]) == 3
    assert len(on.json()["recommendations"]) == 3
