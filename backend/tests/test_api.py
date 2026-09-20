import pytest
from fastapi.testclient import TestClient

from app.main import app, _state


class _StubExplainer:
    """Deterministic, network-free stand-in for ExplanationGenerator so the
    API test suite doesn't depend on internet access or a live Groq key."""

    available = True

    def generate(self, standard, match, certification=None):
        return {
            "explanation": f"stub explanation for {standard.get('number')}",
            "source": "stub",
        }


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        _state["explainer"] = _StubExplainer()
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["standards_loaded"] > 0
    assert body["embedding_model"] == "paraphrase-multilingual-MiniLM-L12-v2"


def test_recommend_happy_path(client):
    resp = client.post("/recommend", json={"text": "structural steel for building construction", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "structural steel for building construction"
    assert len(body["recommendations"]) == 3
    top = body["recommendations"][0]
    for key in ("id", "number", "title", "scope", "match", "version", "certification", "allied_standards"):
        assert key in top
    assert top["match"]["confidence_band"] in ("high", "medium", "low")


def test_recommend_rejects_empty_text(client):
    resp = client.post("/recommend", json={"text": "   ", "top_k": 3})
    assert resp.status_code == 400


def test_recommend_hindi_query_does_not_crash_and_is_relevant(client):
    resp = client.post(
        "/recommend",
        json={"text": "निर्माण में उपयोग होने वाले स्ट्रक्चरल स्टील के लिए मानक", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["recommendations"]) == 3
    assert body["recommendations"][0]["match"]["lexical_rank"] is None


def test_batch_splits_and_scores_each_line(client):
    doc = "1. Structural steel sections for building frame\n2. Gold jewellery fineness marking"
    resp = client.post("/batch", json={"document_text": doc, "top_k_per_item": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["item_count"] == 2
    assert body["items"][0]["recommendations"][0]["number"]


def test_batch_rejects_document_with_no_extractable_items(client):
    resp = client.post("/batch", json={"document_text": "1.\na)\n", "top_k_per_item": 1})
    assert resp.status_code == 400


def test_standards_list_and_department_filter(client):
    resp = client.get("/standards")
    assert resp.status_code == 200
    all_count = resp.json()["count"]
    assert all_count > 0

    resp2 = client.get("/standards", params={"department": "Civil Engineering"})
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["count"] > 0
    assert body2["count"] <= all_count
    assert all(s["department"] == "Civil Engineering" for s in body2["standards"])


def test_qco_orders_list(client):
    resp = client.get("/qco-orders")
    assert resp.status_code == 200
    assert resp.json()["count"] > 0


def test_get_standard_by_id(client):
    resp = client.get("/standard/IS_2062_2011")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "IS_2062_2011"
    assert "allied_standards" in body
    assert "certification" in body
    assert "version" in body


def test_get_unknown_standard_returns_404(client):
    resp = client.get("/standard/NOT_A_REAL_ID")
    assert resp.status_code == 404


def test_lint_flags_superseded_and_missing_marking(client):
    draft = (
        "1. Supply of hot rolled structural steel sections conforming to IS 2062:2006.\n"
        "2. Packaged mineral water bottles as per IS 14543:2004.\n"
    )
    resp = client.post("/lint", json={"document_text": draft, "suggest_missing": False})
    assert resp.status_code == 200
    body = resp.json()

    assert body["citations_found"] == 2
    rules = {f["rule_id"] for f in body["findings"]}
    assert "superseded_citation" in rules
    assert body["summary"]["high"] >= 1

    superseded = [f for f in body["findings"] if f["rule_id"] == "superseded_citation"]
    assert {f["standard_id"] for f in superseded} == {"IS_2062_2006", "IS_14543_2004"}


def test_lint_suggests_standards_for_uncited_items(client):
    draft = "1. Supply of protective helmets for two-wheeler riders, 200 nos."
    resp = client.post("/lint", json={"document_text": draft})
    assert resp.status_code == 200
    body = resp.json()

    uncited = [f for f in body["findings"] if f["rule_id"] == "uncited_item"]
    assert len(uncited) == 1
    assert uncited[0]["standard_id"] == "IS_4151_2015"


def test_lint_spans_index_into_submitted_text(client):
    draft = "Steel shall conform to IS 2062:2006 throughout."
    resp = client.post("/lint", json={"document_text": draft, "suggest_missing": False})
    finding = next(f for f in resp.json()["findings"] if f["rule_id"] == "superseded_citation")
    start, end = finding["span"]
    assert draft[start:end] == "IS 2062:2006"


def test_lint_clean_document(client):
    draft = (
        "Steel shall conform to IS 2062:2011, tensile tested per IS 1608, "
        "erected per IS 800. All material shall be ISI marked."
    )
    resp = client.post("/lint", json={"document_text": draft, "suggest_missing": False})
    assert resp.status_code == 200
    assert resp.json()["findings"] == []


def test_lint_rejects_empty_document(client):
    resp = client.post("/lint", json={"document_text": "   "})
    assert resp.status_code == 400


def test_explain_endpoint(client):
    resp = client.post(
        "/explain",
        json={
            "standard_id": "IS_2062_2011",
            "confidence_band": "high",
            "semantic_similarity": 0.81,
            "lexical_rank": 1,
            "semantic_rank": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["standard_id"] == "IS_2062_2011"
    assert "explanation" in body


def test_explain_unknown_standard_returns_404(client):
    resp = client.post("/explain", json={"standard_id": "NOT_A_REAL_ID"})
    assert resp.status_code == 404
