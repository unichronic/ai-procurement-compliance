import pytest

from app.core.limits import MAX_DOCUMENT_CHARS, MAX_QUERY_CHARS, RateLimiter
from app.main import _confidence_band


def test_allows_up_to_limit_then_blocks():
    limiter = RateLimiter(limit=3, window_seconds=60)
    assert [limiter.check("1.2.3.4", now=1000) for _ in range(3)] == [True, True, True]
    assert limiter.check("1.2.3.4", now=1000) is False


def test_limit_is_per_caller():
    limiter = RateLimiter(limit=2, window_seconds=60)
    limiter.check("a", now=1000)
    limiter.check("a", now=1000)
    assert limiter.check("a", now=1000) is False
    assert limiter.check("b", now=1000) is True


def test_window_expiry_releases_quota():
    limiter = RateLimiter(limit=2, window_seconds=60)
    limiter.check("a", now=1000)
    limiter.check("a", now=1000)
    assert limiter.check("a", now=1030) is False   # still inside the window
    assert limiter.check("a", now=1061) is True    # window rolled over


def test_retry_after_is_positive_when_throttled():
    limiter = RateLimiter(limit=1, window_seconds=60)
    limiter.check("a", now=1000)
    assert 1 <= limiter.retry_after("a", now=1010) <= 60


@pytest.mark.parametrize("similarity,data_confidence,expected", [
    (0.80, "high", "high"),
    (0.40, "high", "medium"),
    (0.10, "high", "low"),
    # A strong match against an unverified record is not a strong recommendation.
    (0.80, "medium", "medium"),
    (0.80, "low", "medium"),
    (0.40, "low", "low"),
])
def test_confidence_band_capped_by_data_confidence(similarity, data_confidence, expected):
    assert _confidence_band(similarity, data_confidence) == expected


def test_recommend_rejects_oversized_query(client):
    resp = client.post("/recommend", json={"text": "x" * (MAX_QUERY_CHARS + 1), "top_k": 3})
    assert resp.status_code == 413


def test_lint_rejects_oversized_document(client):
    resp = client.post("/lint", json={"document_text": "x" * (MAX_DOCUMENT_CHARS + 1)})
    assert resp.status_code == 413


def test_batch_rejects_oversized_document(client):
    resp = client.post("/batch", json={"document_text": "x" * (MAX_DOCUMENT_CHARS + 1)})
    assert resp.status_code == 413


def test_recommendation_reports_source_verification(client):
    resp = client.post("/recommend", json={"text": "drinking water quality", "top_k": 5})
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    assert recs
    assert all(isinstance(r["verified_against_source"], bool) for r in recs)


def test_pipeline_ingested_record_is_flagged_verified(client):
    """Asserted by id rather than by search: records the pipeline verified
    against the real BIS document are now a fraction of a percent of the
    corpus, so whether one lands in an arbitrary top-5 says nothing."""
    resp = client.get("/standard/IS_10500_2012")
    assert resp.status_code == 200
    assert resp.json().get("source"), "pipeline-ingested record lost its source provenance"
