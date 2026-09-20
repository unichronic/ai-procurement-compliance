from app.core.versioning import VersionResolver


def test_active_standard_is_current(sample_standards):
    resolver = VersionResolver(sample_standards)
    result = resolver.resolve("IS_2062_2011")
    assert result["is_current"] is True
    assert result["status"] == "active"
    assert result["warning"] is None
    assert result["latest_active_ids"] == ["IS_2062_2011"]


def test_superseded_standard_resolves_to_latest(sample_standards):
    resolver = VersionResolver(sample_standards)
    result = resolver.resolve("IS_2062_2006")
    assert result["is_current"] is False
    assert result["status"] == "superseded"
    assert result["supersession_chain"] == ["IS_2062_2006", "IS_2062_2011"]
    assert result["latest_active_ids"] == ["IS_2062_2011"]
    assert "SUPERSEDED" in result["warning"]
    assert "current edition" in result["warning"]


def test_unknown_standard_id(sample_standards):
    resolver = VersionResolver(sample_standards)
    result = resolver.resolve("NOT_A_REAL_ID")
    assert "error" in result
