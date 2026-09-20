import pytest

from app.core.citation import (
    CitationResolver,
    format_citation,
    has_marking_requirement,
    parse_citations,
)


@pytest.mark.parametrize("text,base,part,section,year", [
    ("conforming to IS 2062", "2062", None, None, None),
    ("as per IS:2062", "2062", None, None, None),
    ("IS 2062-2011 grade E250", "2062", None, None, 2011),
    ("IS 2062 : 2011", "2062", None, None, 2011),
    ("toys per IS 9873(P-4):2017", "9873", "4", None, 2017),
    ("IS 9873 (Part 4)", "9873", "4", None, None),
    ("motor to IS/IEC 60034-1", "60034", "1", None, None),
    ("water per BIS 10500", "10500", None, None, None),
    ("IS 875 (Part 3) wind loads", "875", "3", None, None),
    ("IS 302 (Part 2/Sec 1)", "302", "2", "1", None),
])
def test_parses_real_world_citation_formats(text, base, part, section, year):
    cites = parse_citations(text)
    assert len(cites) == 1
    c = cites[0]
    assert (c["base"], c["part"], c["section"], c["year"]) == (base, part, section, year)


def test_dash_disambiguation_year_vs_part():
    """IS 2062-2011 is a year; IS/IEC 60034-1 is a part. Same separator."""
    assert parse_citations("IS 2062-2011")[0]["year"] == 2011
    assert parse_citations("IS 2062-2011")[0]["part"] is None
    assert parse_citations("IS/IEC 60034-1")[0]["part"] == "1"
    assert parse_citations("IS/IEC 60034-1")[0]["year"] is None


def test_no_false_positives_on_plain_text():
    assert parse_citations("no citation here at all") == []
    assert parse_citations("quantity 2062 units required") == []


def test_spans_point_at_the_citation(sample_standards):
    text = "Steel shall conform to IS 2062:2011 in all respects."
    c = parse_citations(text)[0]
    start, end = c["span"]
    assert text[start:end] == "IS 2062:2011"


def test_finds_multiple_citations():
    text = "Per IS 2062:2011 and IS 1608, tested to IS 800."
    assert [c["base"] for c in parse_citations(text)] == ["2062", "1608", "800"]


def test_resolve_year_specific_citation_returns_that_edition(sample_standards):
    resolver = CitationResolver(sample_standards)
    c = parse_citations("IS 2062:2006")[0]
    resolved = resolver.resolve(c)
    assert resolved["id"] == "IS_2062_2006"
    assert resolved["status"] == "superseded"


def test_resolve_bare_number_prefers_active_edition(sample_standards):
    resolver = CitationResolver(sample_standards)
    resolved = resolver.resolve(parse_citations("IS 2062")[0])
    assert resolved["id"] == "IS_2062_2011"


def test_resolve_unknown_number_returns_none(sample_standards):
    resolver = CitationResolver(sample_standards)
    assert resolver.resolve(parse_citations("IS 99999")[0]) is None


def test_format_citation_plain(sample_standards):
    steel = next(s for s in sample_standards if s["id"] == "IS_2062_2011")
    assert format_citation(steel) == "IS 2062:2011"


def test_format_citation_includes_latest_amendment(sample_standards):
    steel = next(s for s in sample_standards if s["id"] == "IS_2062_2011")
    steel["amendments"] = [
        {"no": 1, "date": "2012-03", "summary": "x"},
        {"no": 2, "date": "2019-08", "summary": "y"},
    ]
    assert format_citation(steel) == "IS 2062:2011, incorporating Amendment No. 2 (2019-08)"


@pytest.mark.parametrize("text,expected", [
    ("The product shall be ISI marked.", True),
    ("Shall bear a valid BIS certification.", True),
    ("Hallmarked gold only.", True),
    ("CRS registration required.", True),
    ("Supply of steel sections, 10 tonnes.", False),
])
def test_marking_requirement_detection(text, expected):
    assert has_marking_requirement(text) is expected
