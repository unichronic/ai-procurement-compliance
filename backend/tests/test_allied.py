from app.core.allied import AlliedGraph, ALLIED_TYPES


def test_all_six_categories_always_present(sample_standards):
    graph = AlliedGraph(sample_standards)
    grouped = graph.grouped_allied("IS_800_2007")
    assert set(grouped.keys()) == set(ALLIED_TYPES)


def test_forward_allied_relations(sample_standards):
    graph = AlliedGraph(sample_standards)
    grouped = graph.grouped_allied("IS_2062_2011")
    assert [e["id"] for e in grouped["test_method"]] == ["IS_1608_2005"]
    assert [e["id"] for e in grouped["installation"]] == ["IS_800_2007"]
    assert grouped["test_method"][0]["direction"] == "forward"


def test_inverse_allied_relations(sample_standards):
    """A query landing on the allied (child) standard should still surface
    the parent that references it, labelled as the inverse relation."""
    graph = AlliedGraph(sample_standards)
    grouped = graph.grouped_allied("IS_1608_2005")
    assert [e["id"] for e in grouped["test_method"]] == ["IS_2062_2011"]
    assert grouped["test_method"][0]["direction"] == "inverse"

    grouped_installation = graph.grouped_allied("IS_800_2007")
    assert [e["id"] for e in grouped_installation["installation"]] == ["IS_2062_2011"]
    assert grouped_installation["installation"][0]["direction"] == "inverse"


def test_no_allied_relations_returns_empty_lists(sample_standards):
    graph = AlliedGraph(sample_standards)
    grouped = graph.grouped_allied("IS_1417")
    assert all(v == [] for v in grouped.values())
