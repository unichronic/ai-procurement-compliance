"""Operational surface: health semantics, request correlation, error handling."""


def test_liveness_does_not_depend_on_the_index(client):
    """Liveness must not fail while the corpus loads. Kubernetes restarts a
    container that fails liveness, so coupling it to a 50s index build would
    restart the pod forever and it would never finish starting."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readiness_reports_ready_once_index_is_built(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["standards_loaded"] > 0


def test_readiness_returns_503_before_the_index_exists(client):
    """A rolling deploy must not route traffic to a process that would 500."""
    from app.main import _state

    index = _state.pop("index")
    try:
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"
    finally:
        _state["index"] = index


def test_every_response_carries_a_request_id(client):
    resp = client.get("/health/live")
    assert resp.headers.get("X-Request-ID")


def test_client_supplied_request_id_is_preserved(client):
    """So a finding can be traced back to the request that produced it, across
    a proxy that already assigned an id."""
    resp = client.get("/health/live", headers={"X-Request-ID": "trace-me-123"})
    assert resp.headers["X-Request-ID"] == "trace-me-123"


def test_unknown_route_is_a_clean_404(client):
    assert client.get("/no-such-endpoint").status_code == 404
