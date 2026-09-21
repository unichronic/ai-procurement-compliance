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


def test_active_edition_outranks_superseded(sample_standards):
    """Superseded editions share number/title/scope with the current one and
    score nearly identically -- returning the outdated one first is the exact
    failure this project exists to prevent."""
    idx = _build(sample_standards)
    hits = idx.search("structural steel specification", top_k=5)
    ids = [h["standard"]["id"] for h in hits]
    assert ids.index("IS_2062_2011") < ids.index("IS_2062_2006")


def test_superseded_demoted_even_when_it_scores_higher(sample_standards):
    """Regression: this was a tie-break, which only fires on an exact score
    match. It held at 39 standards and silently stopped working at 6,383, where
    slightly different lexical ranks let a superseded edition win outright.

    The superseded fixture is worded to beat the active one on its own terms --
    the demotion has to hold anyway."""
    idx = _build(sample_standards)
    hits = idx.search(
        "Earlier edition of structural steel specification revised and reissued",
        top_k=5,
    )
    ids = [h["standard"]["id"] for h in hits]
    assert ids.index("IS_2062_2011") < ids.index("IS_2062_2006")


def test_superseded_still_retrievable(sample_standards):
    """Demoted, not removed -- the linter has to recognise a superseded edition
    when a draft cites one."""
    idx = _build(sample_standards)
    ids = [h["standard"]["id"] for h in idx.search("structural steel", top_k=5)]
    assert "IS_2062_2006" in ids


def test_model_name_and_size(sample_standards):
    idx = _build(sample_standards)
    assert idx.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
    assert idx.size == len(sample_standards)


def test_embedding_cache_round_trip(sample_standards, tmp_path, monkeypatch):
    """A warm start must reuse embeddings: encoding 6,383 documents takes ~50s
    and would otherwise be paid on every boot and every test session."""
    import app.core.retrieval as retrieval
    monkeypatch.setattr(retrieval, "EMBEDDING_CACHE_DIR", tmp_path)

    cold = retrieval.StandardsIndex(sample_standards)
    cold.build()
    assert cold.cache_hit is False
    assert list(tmp_path.glob("*.npz")), "cold build wrote no cache"

    warm = retrieval.StandardsIndex(sample_standards)
    warm.build()
    assert warm.cache_hit is True
    assert warm.search("structural steel", top_k=3), "warm index must still answer"


def test_cache_key_changes_when_corpus_changes(sample_standards, tmp_path, monkeypatch):
    """Keyed on content, so an edit invalidates it with no manual clear step."""
    import app.core.retrieval as retrieval
    monkeypatch.setattr(retrieval, "EMBEDDING_CACHE_DIR", tmp_path)

    retrieval.StandardsIndex(sample_standards).build()
    edited = [dict(s) for s in sample_standards]
    edited[0]["scope"] = edited[0]["scope"] + " Additional scope text."

    assert retrieval.StandardsIndex(edited).build() is None
    assert len(list(tmp_path.glob("*.npz"))) == 2, "edited corpus reused a stale cache"


def test_corrupt_cache_does_not_break_startup(sample_standards, tmp_path, monkeypatch):
    import app.core.retrieval as retrieval
    monkeypatch.setattr(retrieval, "EMBEDDING_CACHE_DIR", tmp_path)

    idx = retrieval.StandardsIndex(sample_standards)
    idx.build()
    for f in tmp_path.glob("*.npz"):
        f.write_bytes(b"not a real npz file")

    recovered = retrieval.StandardsIndex(sample_standards)
    recovered.build()
    assert recovered.cache_hit is False
    assert recovered.search("structural steel", top_k=2)
