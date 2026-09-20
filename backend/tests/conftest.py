import os
import sys
from pathlib import Path

# Every TestClient request shares one client host, so the production rate limit
# would throttle the suite itself. Raised before app import; the limiter's own
# behaviour is tested directly in test_limits.py.
os.environ.setdefault("RATE_LIMIT_REQUESTS", "1000000")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for data_pipeline

import pytest

STEEL = {
    "id": "IS_2062_2011",
    "number": "IS 2062",
    "edition_year": 2011,
    "title": "Hot Rolled Medium and High Tensile Structural Steel — Specification",
    "scope": "Prescribes chemical composition, mechanical properties, and dimensional tolerances for structural steel used in construction and fabrication.",
    "committee": "MTD 4",
    "department": "Metallurgical Engineering",
    "status": "active",
    "superseded_by": None,
    "supersedes": ["IS_2062_2006"],
    "amendments": [],
    "allied": [
        {"id": "IS_1608_2005", "type": "test_method", "note": "Tensile testing method referenced for mechanical property verification."},
        {"id": "IS_800_2007", "type": "installation", "note": "Referenced for design/fabrication use of the steel grade in general construction."},
    ],
    "qco": {"mandatory": True, "order_id": "QCO_STEEL_2021", "scheme": "ISI"},
    "data_confidence": "high",
    "aliases": ["MS plate", "mild steel plate", "structural steel", "MS angle"],
    "designations": ["E250", "E350", "Fe 410"],
}

STEEL_OLD = {
    "id": "IS_2062_2006",
    "number": "IS 2062",
    "edition_year": 2006,
    "title": "Hot Rolled Low, Medium and High Tensile Structural Steel — Specification",
    "scope": "Earlier edition of structural steel specification, revised and reissued as IS 2062:2011.",
    "committee": "MTD 4",
    "department": "Metallurgical Engineering",
    "status": "superseded",
    "superseded_by": ["IS_2062_2011"],
    "supersedes": None,
    "amendments": [],
    "allied": [],
    "qco": None,
    "data_confidence": "medium",
    "aliases": ["MS plate", "structural steel"],
    "designations": ["Fe 410"],
}

GOLD = {
    "id": "IS_1417",
    "number": "IS 1417",
    "edition_year": 2016,
    "title": "Grading and Fineness of Gold and Gold Alloys, and Their Fineness Marking — Specification",
    "scope": "Specifies grading and fineness marking requirements for gold and gold alloys.",
    "committee": "MTD 24",
    "department": "Metallurgical Engineering",
    "status": "active",
    "superseded_by": None,
    "supersedes": None,
    "amendments": [],
    "allied": [],
    "qco": {"mandatory": False, "order_id": None, "scheme": "Hallmark", "note": "Hallmarking is mandatory for gold; voluntary for silver as of 2026."},
    "data_confidence": "high",
    "aliases": ["gold jewellery", "gold hallmark", "sona"],
    "designations": ["22K", "916"],
}

TEST_METHOD = {
    "id": "IS_1608_2005",
    "number": "IS 1608",
    "edition_year": 2005,
    "title": "Metallic Materials — Tensile Testing at Ambient Temperature",
    "scope": "Specifies the method for tensile testing of metallic materials at ambient temperature.",
    "committee": "MTD 3",
    "department": "Metallurgical Engineering",
    "status": "active",
    "superseded_by": None,
    "supersedes": None,
    "amendments": [],
    "allied": [],
    "qco": None,
    "data_confidence": "high",
}

INSTALLATION = {
    "id": "IS_800_2007",
    "number": "IS 800",
    "edition_year": 2007,
    "title": "General Construction in Steel — Code of Practice",
    "scope": "Covers design, fabrication, and erection of general steel construction.",
    "committee": "CED 7",
    "department": "Civil Engineering",
    "status": "active",
    "superseded_by": None,
    "supersedes": None,
    "amendments": [],
    "allied": [],
    "qco": None,
    "data_confidence": "medium",
}

SAMPLE_STANDARDS = [STEEL, STEEL_OLD, GOLD, TEST_METHOD, INSTALLATION]

SAMPLE_QCO_ORDERS = [
    {
        "id": "QCO_STEEL_2021",
        "name": "Steel and Steel Products (Quality Control) Order, 2021 (Amendment)",
        "issuing_authority": "Ministry of Steel",
        "scheme": "ISI",
        "notified": "2021-01",
    }
]


class StubExplainer:
    """Deterministic, network-free stand-in for ExplanationGenerator so the API
    tests need no internet access or live Groq key."""

    available = True

    def generate(self, standard, match, certification=None):
        return {
            "explanation": f"stub explanation for {standard.get('number')}",
            "source": "stub",
        }


@pytest.fixture(scope="session")
def client():
    """Shared across the whole session -- app startup builds the real embedding
    index, which is far too expensive to repeat per module."""
    from fastapi.testclient import TestClient
    from app.main import app, _state

    with TestClient(app) as c:
        _state["explainer"] = StubExplainer()
        yield c


@pytest.fixture
def sample_standards():
    import copy
    return copy.deepcopy(SAMPLE_STANDARDS)


@pytest.fixture
def sample_qco_orders():
    import copy
    return copy.deepcopy(SAMPLE_QCO_ORDERS)
