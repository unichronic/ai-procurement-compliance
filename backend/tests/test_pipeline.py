"""Pipeline tests. All offline -- network lives behind mirror._get, and every
parser here is a pure function fed a fixture resembling real scanned BIS text.
"""
import pytest

from data_pipeline import corpus
from data_pipeline.extract import (
    build_chunks,
    extract_committee,
    extract_ics,
    extract_referred_standards,
    extract_scope,
    extract_standard,
    extract_title,
)
from data_pipeline.mirror import parse_catalogue_html, parse_item_id, text_url

SAMPLE_DOC = """IS 10500 : 2012

Hkkjrh; ekud

Indian Standard DRINKING WATER -- SPECIFICATION ( Second Revision )

ICS 13.060.20

BUREAU OF INDIAN STANDARDS
MANAK BHAVAN, 9 BAHADUR SHAH ZAFAR MARG NEW DELHI 110002

Drinking Water Sectional Committee, FAD 25

FOREWORD This Indian Standard was adopted by the Bureau of Indian Standards.

1 SCOPE This standard prescribes the requirements and the methods of sampling
and test for drinking water intended for human consumption.
2 REFERENCES The standards listed in Annex A contain provisions which through
reference in this text constitute provisions of this standard.
4 REQUIREMENTS Drinking water shall comply with the requirements given in
Table 1 and Table 2 when tested in accordance with the methods specified.
The acceptable limit for turbidity shall be 1 NTU and the permissible limit in
the absence of an alternate source shall be 5 NTU as specified herein.

ANNEX A (Clause 2)
LIST OF REFERRED INDIAN STANDARDS IS No. 1622 : 1981 Title Methods of sampling
15302 : 2002 13428 : 2003

3025

(Part 1) : 1987 (Part 2) : 2002 (Part 4) : 1983
"""


def test_parse_item_id_simple():
    assert parse_item_id("gov.in.is.10500.2012") == {
        "item_id": "gov.in.is.10500.2012", "number": "10500",
        "part": None, "section": None, "year": 2012,
    }


def test_parse_item_id_with_part():
    parsed = parse_item_id("gov.in.is.9873.4.2017")
    assert (parsed["number"], parsed["part"], parsed["year"]) == ("9873", "4", 2017)


def test_parse_item_id_rejects_malformed():
    assert parse_item_id("gov.in.is.notanumber") is None
    assert parse_item_id("gov.in.is.10500") is None  # no year
    assert parse_item_id("something.else.entirely") is None


def test_text_url_construction():
    assert text_url("gov.in.is.10500.2012") == \
        "https://archive.org/download/gov.in.is.10500.2012/is.10500.2012.txt"


def test_parse_catalogue_html_dedupes():
    html = """
      <a href="https://archive.org/details/gov.in.is.10.1.1990">x</a>
      <a href="https://archive.org/download/gov.in.is.10.1.1990/is.10.1.1990.txt">t</a>
      <a href="https://archive.org/details/gov.in.is.190.1991">y</a>
    """
    assert parse_catalogue_html(html) == ["gov.in.is.10.1.1990", "gov.in.is.190.1991"]


def test_extract_basic_fields():
    assert extract_title(SAMPLE_DOC) == "Drinking Water — Specification"
    assert extract_ics(SAMPLE_DOC) == "13.060.20"
    assert extract_committee(SAMPLE_DOC) == "FAD 25"
    assert "methods of sampling" in extract_scope(SAMPLE_DOC)


def test_title_case_normalization_preserves_acronyms():
    doc = SAMPLE_DOC.replace("DRINKING WATER -- SPECIFICATION",
                             "SELF-BALLASTED LED LAMPS AND PVC CABLES")
    title = extract_title(doc)
    assert "LED" in title and "PVC" in title
    assert "Lamps" in title and " and " in title


@pytest.mark.parametrize("junk", ["~", "( Reaffirmed 2002 )", "-- x", "AB"])
def test_junk_titles_are_rejected(junk):
    """Scanned text yields garbage titles; accepting them would overwrite good
    curated data with noise."""
    doc = SAMPLE_DOC.replace("DRINKING WATER -- SPECIFICATION", junk)
    assert extract_title(doc) is None


def test_short_scope_rejected_as_extraction_failure():
    doc = SAMPLE_DOC.replace(
        "This standard prescribes the requirements and the methods of sampling\nand test for drinking water intended for human consumption.",
        "x.")
    assert extract_scope(doc) is None


def test_referred_standards_attaches_parts_to_bare_base():
    refs = extract_referred_standards(SAMPLE_DOC, self_number="10500")
    by_number = {}
    for r in refs:
        by_number.setdefault(r["number"], []).append(r["part"])

    assert "1622" in by_number
    assert sorted(p for p in by_number.get("3025", []) if p) == ["1", "2", "4"]


def test_referred_standards_filters_self_reference():
    refs = extract_referred_standards(SAMPLE_DOC, self_number="10500")
    assert all(r["number"] != "10500" for r in refs)


def test_referred_standards_does_not_invent_parts_for_dated_entries():
    """A part run following a complete dated citation usually belongs to a base
    the page break separated. Dropping beats fabricating a normative reference,
    because a wrong reference becomes a wrong lint finding."""
    doc = SAMPLE_DOC.replace("3025\n", "")
    refs = extract_referred_standards(doc, self_number="10500")
    assert all(not (r["number"] == "13428" and r["part"]) for r in refs)


def test_build_chunks_includes_scope_and_body():
    chunks = build_chunks(SAMPLE_DOC, self_number="10500")
    kinds = [c["kind"] for c in chunks]
    assert "scope" in kinds
    assert "body" in kinds
    assert all(c["text"] for c in chunks)


def test_body_chunks_exclude_boilerplate():
    chunks = build_chunks(SAMPLE_DOC, self_number="10500")
    joined = " ".join(c["text"] for c in chunks)
    assert "MANAK BHAVAN" not in joined
    assert "BUREAU OF INDIAN STANDARDS" not in joined


def test_extract_standard_end_to_end():
    item = parse_item_id("gov.in.is.10500.2012")
    record = extract_standard(SAMPLE_DOC, item=item)
    assert record["id"] == "IS_10500_2012"
    assert record["number"] == "IS 10500"
    assert record["edition_year"] == 2012
    assert record["data_confidence"] == "high"
    assert record["source"]["item_id"] == "gov.in.is.10500.2012"


# -- corpus merge / change detection -----------------------------------------

def _ingested(**over):
    base = {
        "id": "IS_2062_2011", "title": "Ingested Title", "scope": "Ingested scope text.",
        "committee": "MTD 4", "ics": "77.140", "chunks": [{"kind": "scope", "text": "s"}],
        "referred_standards": [{"number": "1608", "part": None, "year": 2005}],
        "source": {"item_id": "gov.in.is.2062.2011"},
    }
    base.update(over)
    return base


def test_merge_preserves_curated_fields(sample_standards):
    existing = next(s for s in sample_standards if s["id"] == "IS_2062_2011")
    merged = corpus.merge_record(existing, _ingested())

    assert merged["qco"] == existing["qco"]
    assert merged["aliases"] == existing["aliases"]
    assert merged["designations"] == existing["designations"]
    assert merged["allied"] == existing["allied"]
    assert merged["status"] == existing["status"]


def test_merge_refreshes_document_fields(sample_standards):
    existing = next(s for s in sample_standards if s["id"] == "IS_2062_2011")
    merged = corpus.merge_record(existing, _ingested())

    assert merged["scope"] == "Ingested scope text."
    assert merged["chunks"] == [{"kind": "scope", "text": "s"}]
    assert merged["data_confidence"] == "high"


def test_merge_never_blanks_curated_data_with_failed_extraction(sample_standards):
    """A thin extraction must not wipe a good curated title."""
    existing = next(s for s in sample_standards if s["id"] == "IS_2062_2011")
    original_title = existing["title"]
    merged = corpus.merge_record(existing, _ingested(title=None, scope=None, chunks=[]))

    assert merged["title"] == original_title
    assert merged["scope"] == existing["scope"]


def test_merge_adds_new_record_with_defaults():
    merged = corpus.merge_record(None, _ingested(id="IS_9999_2020"))
    assert merged["status"] == "active"
    assert merged["aliases"] == []
    assert merged["qco"] is None


def test_change_detection_reports_added_changed_removed():
    records = [_ingested(), _ingested(id="IS_NEW_2020")]
    snapshot = {
        "IS_2062_2011": "differenthash00",
        "IS_GONE_1999": "somehash0000000",
    }
    delta = corpus.diff_against_snapshot(records, snapshot)
    assert delta["added"] == ["IS_NEW_2020"]
    assert delta["changed"] == ["IS_2062_2011"]
    assert delta["removed"] == ["IS_GONE_1999"]


def test_content_hash_ignores_curated_fields():
    a = _ingested(qco={"mandatory": True}, aliases=["x"])
    b = _ingested(qco=None, aliases=[])
    assert corpus.content_hash(a) == corpus.content_hash(b)


def test_content_hash_changes_with_document_content():
    assert corpus.content_hash(_ingested()) != corpus.content_hash(_ingested(scope="different"))


def test_merge_corpus_counts(sample_standards):
    merged, stats = corpus.merge_corpus(sample_standards, [_ingested(), _ingested(id="IS_NEW_2020")])
    assert stats["updated"] == 1
    assert stats["added"] == 1
    assert len(merged) == len(sample_standards) + 1
