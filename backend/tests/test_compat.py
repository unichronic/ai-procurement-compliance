"""The contract the team's React frontend was written against.

frontend/src/api/client.js calls /retrieve, /languages, /extract and
/standards/{id}/... . This engine natively exposes /recommend and /lint, so
without this layer pointing the React app at this engine does nothing.
"""


def test_retrieve_returns_the_frontend_shape(client):
    resp = client.post("/retrieve", json={"query": "structural steel plate", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()

    assert body["confidence"] in ("strong", "uncertain", "none")
    assert len(body["results"]) == 3
    for r in body["results"]:
        for key in ("id", "number", "title", "final_score", "stage_scores",
                    "ranker_used", "status", "certification", "citation"):
            assert key in r, f"missing {key}"
        for key in ("dense", "bm25", "cross_encoder", "ltr_or_fallback"):
            assert key in r["stage_scores"]


def test_final_score_is_monotonic_with_rank(client):
    """Regression: final_score was raw cosine while the ranking used lexical,
    alias and designation channels too, so the UI showed a higher score on a
    lower-ranked row."""
    body = client.post("/retrieve", json={"query": "MS plate for fabrication",
                                          "top_k": 5}).json()
    scores = [r["final_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True), scores


def test_nonsense_query_is_never_reported_as_strong(client):
    """Measured top-1 similarity ranges overlap between real and nonsense
    queries, so the gate cannot be crisp. The property that must hold is that
    nonsense never reads as a confident recommendation."""
    for junk in ("zzqq nonexistent widget", "asdfgh qwerty zxcvb",
                 "purple monkey dishwasher"):
        body = client.post("/retrieve", json={"query": junk, "top_k": 3}).json()
        assert body["confidence"] != "strong", f"{junk!r} reported as strong"


def test_unknown_certification_maps_to_not_verified_never_a_clearance(client):
    """The frontend's scheme enum has a not_verified member; unchecked records
    must land there with mandatory false, not on a 'none' that reads as
    cleared."""
    body = client.post("/retrieve", json={"query": "cement concrete works",
                                          "top_k": 8}).json()
    schemes = {r["certification"]["scheme"] for r in body["results"]}
    assert schemes <= {"ISI", "CRS", "Hallmark", "none", "not_verified"}
    for r in body["results"]:
        cert = r["certification"]
        if cert["scheme"] == "not_verified":
            assert cert["mandatory"] is False
            assert "not" in cert["explanation"].lower()


def test_languages_does_not_claim_a_translation_stage(client):
    body = client.get("/languages").json()
    codes = [l["code"] for l in body["languages"]]
    assert "en" in codes and "hi" in codes
    assert "translation" in body["note"].lower()


def test_standards_list_and_detail_and_related(client):
    listing = client.get("/standards", params={"limit": 5}).json()
    assert listing["count"] > 0 and len(listing["standards"]) == 5

    detail = client.get("/standards/IS_2062_2011")
    assert detail.status_code == 200
    # Full citation form, per their contract ("IS 694:2010"): the detail page
    # renders this directly, and an officer needs the citable designation.
    assert detail.json()["number"] == "IS 2062:2011"

    # Resolvable by bare number and by citation, which is how the UI links.
    assert client.get("/standards/IS%202062").status_code == 200
    assert client.get("/standards/IS%202062:2011").status_code == 200

    related = client.get("/standards/IS_2062_2011/related").json()
    assert related["total"] >= 1
    assert related["researched"] is True
    # Forward dependencies grouped by relation, inverse references separate.
    assert any(g["standards"] for g in related["depends_on"]) or related["referenced_by"]
    for group in related["depends_on"]:
        assert group["heading"]
        assert all("outside_corpus" in x for x in group["standards"])


def test_related_marks_references_outside_the_corpus(client):
    """3,009 extracted references name standards this corpus does not hold.
    Showing them as "cited but not held" is honest; omitting them silently
    would imply the cluster is complete."""
    related = client.get("/standards/IS_10500_2012/related").json()
    groups = {g["type"]: g for g in related["depends_on"]}
    if "outside_corpus" in groups:
        assert all(x["outside_corpus"] for x in groups["outside_corpus"]["standards"])
        assert "nothing is claimed" in groups["outside_corpus"]["explanation"]


def test_related_absence_is_not_a_claim_of_none(client):
    """`researched` distinguishes "no relationships" from "never looked"."""
    related = client.get("/standards/IS_2062_2011/related").json()
    assert "researched" in related


def test_amendments_absence_is_not_a_claim_of_none(client):
    body = client.get("/standards/IS_2062_2011/amendments").json()
    assert "checked" in body, "must distinguish 'no amendments' from 'not checked'"


def test_unknown_standard_is_404(client):
    assert client.get("/standards/IS_NOT_REAL_9999").status_code == 404


def test_extract_accepts_an_upload(client):
    body = client.post(
        "/extract",
        files={"file": ("spec.txt", b"Supply of structural steel plate to IS 2062.",
                        "text/plain")},
        params={"top_k": 3},
    )
    assert body.status_code == 200
    data = body.json()
    assert data["extracted_characters"] > 0
    assert len(data["results"]) == 3
