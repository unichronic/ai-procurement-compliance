import pytest

from app.core.allied import AlliedGraph
from app.core.certification import CertificationAdvisor
from app.core.lint import SpecLinter
from app.core.versioning import VersionResolver


class _StubIndex:
    """Deterministic stand-in for StandardsIndex -- lint rules are about rule
    logic, not embedding quality, and the real index costs a model load."""

    def __init__(self, standards, hit_id=None, similarity=0.8):
        self.by_id = {s["id"]: s for s in standards}
        self._hit_id = hit_id
        self._similarity = similarity

    def search(self, query, top_k=5):
        if not self._hit_id:
            return []
        return [{
            "standard": self.by_id[self._hit_id],
            "semantic_similarity": self._similarity,
            "fused_score": 0.03,
            "lexical_rank": 1,
            "semantic_rank": 1,
            "bm25_score": 1.0,
            "designation_hits": 0,
        }]


def _linter(standards, qco_orders, hit_id=None, similarity=0.8):
    return SpecLinter(
        standards=standards,
        index=_StubIndex(standards, hit_id, similarity),
        versions=VersionResolver(standards),
        allied=AlliedGraph(standards),
        certification=CertificationAdvisor(standards, qco_orders),
    )


def _rules(result):
    return {f["rule_id"] for f in result["findings"]}


def _by_rule(result, rule_id):
    return [f for f in result["findings"] if f["rule_id"] == rule_id]


def test_flags_superseded_citation(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint("Steel shall conform to IS 2062:2006, ISI marked.", suggest_missing=False)

    findings = _by_rule(result, "superseded_citation")
    assert len(findings) == 1
    assert findings[0]["severity"] == "high"
    assert "IS 2062:2011" in findings[0]["suggested_fix"]
    assert findings[0]["standard_id"] == "IS_2062_2006"


def test_does_not_flag_current_edition(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint("Steel shall conform to IS 2062:2011, ISI marked.", suggest_missing=False)
    assert "superseded_citation" not in _rules(result)


def test_flags_missing_normative_reference(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint("Steel to IS 2062:2011, ISI marked.", suggest_missing=False)

    findings = _by_rule(result, "missing_allied_standard")
    assert [f["standard_id"] for f in findings] == ["IS_1608_2005"]
    assert "IS 1608" in findings[0]["suggested_fix"]


def test_no_missing_allied_finding_when_already_cited(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint(
        "Steel to IS 2062:2011 tested per IS 1608. ISI marked.", suggest_missing=False
    )
    assert "missing_allied_standard" not in _rules(result)


def test_flags_missing_mandatory_marking(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint("Supply of steel sections to IS 2062:2011.", suggest_missing=False)

    findings = _by_rule(result, "missing_certification_requirement")
    assert len(findings) == 1
    assert findings[0]["severity"] == "high"
    assert "ISI" in findings[0]["suggested_fix"]


def test_no_marking_finding_when_mark_required(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint(
        "Supply of steel sections to IS 2062:2011. The product shall be ISI marked.",
        suggest_missing=False,
    )
    assert "missing_certification_requirement" not in _rules(result)


def test_no_marking_finding_for_voluntary_scheme(sample_standards, sample_qco_orders):
    """IS 1417 hallmarking is voluntary in the fixture -- must not be asserted
    as mandatory. A linter that always finds something isn't running logic."""
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint("Supply of gold jewellery to IS 1417.", suggest_missing=False)
    assert "missing_certification_requirement" not in _rules(result)


def test_flags_unresolved_citation(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint("Works as per IS 99999.", suggest_missing=False)

    findings = _by_rule(result, "unresolved_citation")
    assert len(findings) == 1
    assert findings[0]["evidence"] == "IS 99999"


def test_flags_uncited_item_with_suggestion(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders, hit_id="IS_2062_2011")
    result = linter.lint("1. Supply of hot rolled steel sections for shed.")

    findings = _by_rule(result, "uncited_item")
    assert len(findings) == 1
    assert findings[0]["standard_id"] == "IS_2062_2011"
    assert "IS 2062:2011" in findings[0]["suggested_fix"]


def test_weak_match_does_not_produce_uncited_finding(sample_standards, sample_qco_orders):
    """Below the similarity floor the engine must stay silent rather than
    assert an applicable standard it isn't confident about."""
    linter = _linter(sample_standards, sample_qco_orders, hit_id="IS_2062_2011", similarity=0.2)
    result = linter.lint("1. Supply of assorted miscellaneous office consumables.")
    assert "uncited_item" not in _rules(result)


def test_item_that_already_cites_is_not_flagged(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders, hit_id="IS_2062_2011")
    result = linter.lint("1. Supply of steel sections to IS 2062:2011, ISI marked.")
    assert "uncited_item" not in _rules(result)


def test_findings_carry_spans_into_source_text(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    text = "Steel shall conform to IS 2062:2006 in all respects."
    result = linter.lint(text, suggest_missing=False)

    finding = _by_rule(result, "superseded_citation")[0]
    start, end = finding["span"]
    assert text[start:end] == "IS 2062:2006"


def test_findings_sorted_high_severity_first(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint("Steel to IS 2062:2006.", suggest_missing=False)
    severities = [f["severity"] for f in result["findings"]]
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


def test_clean_document_produces_no_findings(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint(
        "Steel shall conform to IS 2062:2011, tested per IS 1608, "
        "erected per IS 800. The product shall be ISI marked.",
        suggest_missing=False,
    )
    assert result["findings"] == []
    assert result["summary"]["total"] == 0


def test_summary_counts(sample_standards, sample_qco_orders):
    linter = _linter(sample_standards, sample_qco_orders)
    result = linter.lint("Steel to IS 2062:2006.", suggest_missing=False)
    summary = result["summary"]
    assert summary["total"] == len(result["findings"])
    assert summary["high"] >= 1
