from app.core.explain import ExplanationGenerator


def test_no_api_key_uses_template_and_is_marked_unavailable():
    gen = ExplanationGenerator(api_key="")
    assert gen.available is False

    result = gen.generate(
        {"number": "IS 800", "scope": "Covers design, fabrication, and erection."},
        {"confidence_band": "medium", "semantic_similarity": 0.35},
        {"mandatory_certification": False},
    )
    assert result["source"] == "template"
    assert "IS 800" in result["explanation"]
    assert "medium" in result["explanation"]


def test_template_mentions_mandatory_certification_when_applicable():
    gen = ExplanationGenerator(api_key="")
    result = gen.generate(
        {"number": "IS 2062", "scope": "Structural steel."},
        {"confidence_band": "high", "semantic_similarity": 0.8},
        {"mandatory_certification": True, "scheme": "ISI"},
    )
    assert "ISI" in result["explanation"]
    assert "mandatory" in result["explanation"].lower()


def test_groq_failure_falls_back_to_template_gracefully(mocker):
    """The explanation layer must never break a request just because the
    Groq API is unreachable, rate-limited, or the key is bad."""
    gen = ExplanationGenerator(api_key="fake-key-for-test")
    assert gen.available is True

    mocker.patch.object(
        gen._client.chat.completions, "create", side_effect=RuntimeError("network down")
    )

    result = gen.generate(
        {"number": "IS 800", "scope": "Covers design, fabrication, and erection."},
        {"confidence_band": "medium", "semantic_similarity": 0.35},
        None,
    )
    assert result["source"] == "template"
    assert "fallback_reason" in result


def test_groq_success_path_is_used_when_available(mocker):
    gen = ExplanationGenerator(api_key="fake-key-for-test")

    class FakeMessage:
        content = "  This standard is a strong match because ...  "

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    mocker.patch.object(gen._client.chat.completions, "create", return_value=FakeResponse())

    result = gen.generate(
        {"number": "IS 2062", "scope": "Structural steel."},
        {"confidence_band": "high", "semantic_similarity": 0.8},
        {"mandatory_certification": True, "scheme": "ISI"},
    )
    assert result["source"] == "groq"
    assert result["explanation"] == "This standard is a strong match because ..."


def test_never_sends_raw_query_text(mocker):
    """Privacy contract: the payload sent to Groq must be built only from
    standard metadata + numeric match signals, never a free-text query."""
    gen = ExplanationGenerator(api_key="fake-key-for-test")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        class R:
            class choices:
                pass
        raise RuntimeError("stop after capture")

    mocker.patch.object(gen._client.chat.completions, "create", side_effect=fake_create)

    gen.generate(
        {"number": "IS 800", "scope": "Covers design, fabrication, and erection.", "title": "x"},
        {"confidence_band": "medium", "semantic_similarity": 0.35},
        {"mandatory_certification": False},
    )

    sent_text = " ".join(m["content"] for m in captured["messages"])
    assert "query" not in sent_text.lower()
