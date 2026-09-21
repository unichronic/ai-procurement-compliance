from app.core.certification import CertificationAdvisor


def test_mandatory_certification(sample_standards, sample_qco_orders):
    advisor = CertificationAdvisor(sample_standards, sample_qco_orders)
    result = advisor.advise("IS_2062_2011")
    assert result["mandatory_certification"] is True
    assert result["scheme"] == "ISI"
    assert result["governing_order"]["id"] == "QCO_STEEL_2021"
    assert "Mandatory" in result["message"]


def test_voluntary_certification(sample_standards, sample_qco_orders):
    advisor = CertificationAdvisor(sample_standards, sample_qco_orders)
    result = advisor.advise("IS_1417")
    assert result["mandatory_certification"] is False
    assert result["scheme"] == "Hallmark"
    assert "scheme_description" in result


def test_no_qco_identified_when_record_was_checked(sample_standards, sample_qco_orders):
    advisor = CertificationAdvisor(sample_standards, sample_qco_orders)
    result = advisor.advise("IS_800_2007")
    assert result["mandatory_certification"] is False
    assert result["certification_status"] == "no_qco_identified"
    assert result["scheme"] is None


def test_unchecked_record_reports_unknown_not_a_clearance(sample_standards, sample_qco_orders):
    """The safety property of this module. QCO data covers a fraction of a
    percent of the corpus, so for almost every standard nobody has looked.
    Reporting that as "certification not required" could strip a mandatory ISI
    requirement out of a live tender."""
    unchecked = dict(sample_standards[0])
    unchecked["id"] = "IS_UNCHECKED_2020"
    unchecked.pop("qco", None)
    unchecked.pop("qco_checked", None)

    advisor = CertificationAdvisor(sample_standards + [unchecked], sample_qco_orders)
    result = advisor.advise("IS_UNCHECKED_2020")

    assert result["certification_status"] == "unknown"
    assert result["mandatory_certification"] is None, \
        "unknown must not be reported as False -- that reads as a clearance"
    assert "not a clearance" in result["message"].lower()


def test_unknown_is_never_falsey_in_a_way_that_reads_as_compliant(
        sample_standards, sample_qco_orders):
    """Guards the specific bug shape: `if not result["mandatory_certification"]`
    treats unknown and no-QCO identically. They are not the same claim."""
    unchecked = dict(sample_standards[0])
    unchecked["id"] = "IS_UNCHECKED_2021"
    unchecked.pop("qco", None)
    unchecked.pop("qco_checked", None)

    advisor = CertificationAdvisor(sample_standards + [unchecked], sample_qco_orders)
    unknown = advisor.advise("IS_UNCHECKED_2021")
    checked = advisor.advise("IS_800_2007")

    assert unknown["certification_status"] != checked["certification_status"]


def test_unknown_standard_id(sample_standards, sample_qco_orders):
    advisor = CertificationAdvisor(sample_standards, sample_qco_orders)
    result = advisor.advise("NOT_A_REAL_ID")
    assert "error" in result
