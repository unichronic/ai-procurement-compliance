from app.core.batch import split_document, split_document_with_spans


def test_splits_numbered_list():
    text = "1. Structural steel sections\n2. Gold jewellery fineness marking\n3. LPG cylinder valve assembly"
    items = split_document(text)
    assert items == [
        "Structural steel sections",
        "Gold jewellery fineness marking",
        "LPG cylinder valve assembly",
    ]


def test_splits_bullet_and_dash_lists():
    text = "- Steel wire ropes\n* Hallmarked gold jewellery\n• LPG valves"
    items = split_document(text)
    assert items == ["Steel wire ropes", "Hallmarked gold jewellery", "LPG valves"]


def test_ignores_blank_lines_and_bare_bullet_markers():
    # Bare markers like "a)" and "1." strip down to nothing and must be dropped,
    # not just left blank -- min length filter guards against noise line items.
    text = "\n\nStructural steel\n\na)\n1.\nAB\n"
    items = split_document(text)
    assert items == ["Structural steel"]


def test_empty_document_returns_no_items():
    assert split_document("") == []
    assert split_document("\n\n   \n") == []


def test_spans_locate_items_in_original_text():
    text = "1. Structural steel sections\n2. Gold jewellery marking"
    items = split_document_with_spans(text)
    assert len(items) == 2
    for cleaned, start, end in items:
        assert text[start:end] == cleaned


def test_spans_skip_the_bullet_marker():
    text = "  - Steel wire ropes"
    (cleaned, start, end), = split_document_with_spans(text)
    assert cleaned == "Steel wire ropes"
    assert text[start:end] == "Steel wire ropes"


def test_spans_are_monotonically_increasing():
    text = "1. Alpha item here\n\n2. Beta item here\n3. Gamma item here"
    items = split_document_with_spans(text)
    starts = [start for _, start, _ in items]
    assert starts == sorted(starts)


def test_lettered_and_parenthetical_prefixes():
    text = "(a) Steel wire ropes\n(b) Gold jewellery"
    items = split_document(text)
    assert items == ["Steel wire ropes", "Gold jewellery"]
