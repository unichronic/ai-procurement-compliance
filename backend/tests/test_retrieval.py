from app.core.retrieval import StandardsIndex, _tokenize


def _build(sample_standards):
    idx = StandardsIndex(sample_standards)
    idx.build()
    return idx


def test_tokenize_ascii():
    assert _tokenize("IS 2062 Steel-Ropes!") == ["is", "2062", "steel", "ropes"]


def test_tokenize_non_latin_script_yields_no_tokens():
    # Devanagari has no [a-zA-Z0-9] characters, so BM25 sees an empty query.
    assert _tokenize("निर्माण स्टील") == []


def test_search_returns_top_k(sample_standards):
    idx = _build(sample_standards)
    hits = idx.search("structural steel for construction", top_k=2)
    assert len(hits) == 2
    assert all("standard" in h and "fused_score" in h for h in hits)


def test_search_ranks_relevant_standard_first_english(sample_standards):
    idx = _build(sample_standards)
    hits = idx.search("structural steel for construction", top_k=3)
    assert hits[0]["standard"]["id"] == "IS_2062_2011"
    assert hits[0]["lexical_rank"] is not None  # English query has real lexical overlap


def test_search_hindi_query_still_ranks_correctly(sample_standards):
    """Regression test: a non-Latin-script query used to get corrupted by BM25
    argsort tie-breaking (document array order leaking in as a fake lexical
    signal) and could outrank the correct semantic top hit. See retrieval.py."""
    idx = _build(sample_standards)
    hits = idx.search("निर्माण में उपयोग होने वाले स्ट्रक्चरल स्टील के लिए मानक", top_k=3)
    assert hits[0]["standard"]["id"] == "IS_2062_2011"
    assert hits[0]["semantic_similarity"] > 0.5
    # No lexical signal at all for this query -- must not be reported as a real rank.
    assert hits[0]["lexical_rank"] is None


def test_search_out_of_vocabulary_query_does_not_crash(sample_standards):
    idx = _build(sample_standards)
    hits = idx.search("zzqqxxnonexistentvocabularyterm", top_k=3)
    assert len(hits) == 3
    assert hits[0]["lexical_rank"] is None


def test_alias_makes_trade_name_findable(sample_standards):
    """'MS plate' shares no vocabulary with 'Hot Rolled ... Structural Steel'.
    Trade-name queries are the realistic case and only work via aliases."""
    idx = _build(sample_standards)
    hits = idx.search("MS plate", top_k=3)
    assert hits[0]["standard"]["id"] in ("IS_2062_2011", "IS_2062_2006")


def test_romanized_hindi_alias_is_matched(sample_standards):
    idx = _build(sample_standards)
    hits = idx.search("sona hallmark", top_k=3)
    assert hits[0]["standard"]["id"] == "IS_1417"


def test_designation_hit_promotes_standard(sample_standards):
    idx = _build(sample_standards)
    hits = idx.search("material of E250 grade", top_k=3)
    assert hits[0]["standard"]["id"] == "IS_2062_2011"
    assert hits[0]["designation_hits"] >= 1


def test_no_designation_hits_reported_when_absent(sample_standards):
    idx = _build(sample_standards)
    hits = idx.search("structural steel for construction", top_k=3)
    assert hits[0]["designation_hits"] == 0


def test_active_edition_outranks_superseded_on_tie(sample_standards):
    """Superseded editions share number/title/scope with the current one and
    score nearly identically -- returning the outdated one first is the exact
    failure this project exists to prevent."""
    idx = _build(sample_standards)
    hits = idx.search("structural steel specification", top_k=5)
    ids = [h["standard"]["id"] for h in hits]
    assert ids.index("IS_2062_2011") < ids.index("IS_2062_2006")


def test_model_name_and_size(sample_standards):
    idx = _build(sample_standards)
    assert idx.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
    assert idx.size == len(sample_standards)
