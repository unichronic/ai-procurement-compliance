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


def test_no_certification_required(sample_standards, sample_qco_orders):
    advisor = CertificationAdvisor(sample_standards, sample_qco_orders)
    result = advisor.advise("IS_800_2007")
    assert result["mandatory_certification"] is False
    assert result["scheme"] is None
    assert "No Quality Control Order" in result["message"]


def test_unknown_standard_id(sample_standards, sample_qco_orders):
    advisor = CertificationAdvisor(sample_standards, sample_qco_orders)
    result = advisor.advise("NOT_A_REAL_ID")
    assert "error" in result
