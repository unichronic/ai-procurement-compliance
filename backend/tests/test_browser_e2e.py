"""Real browser end-to-end tests.

Everything else verifies the API contract; these verify the thing a user
actually touches. They drive headless Chromium against a live server, so they
catch what contract tests structurally cannot: a JS exception, an element that
never renders, a handler wired to the wrong id.

Skipped automatically when Playwright or its browser binary isn't installed,
so the suite still runs anywhere:

    pip install playwright && playwright install chromium
"""
import socket
import threading
import time

import pytest

playwright_api = pytest.importorskip(
    "playwright.sync_api", reason="playwright not installed"
)

from app.main import app  # noqa: E402
from tests.conftest import StubExplainer  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    import uvicorn
    from app.main import _state

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 300
    while not server.started and time.time() < deadline:
        time.sleep(0.2)
    if not server.started:
        pytest.skip("server did not start in time")

    _state["explainer"] = StubExplainer()  # no network in tests
    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=30)


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # binary not installed
            pytest.skip(f"chromium unavailable: {exc}")
        yield b
        b.close()


@pytest.fixture
def page(browser, live_server):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{live_server}/ui/", wait_until="load")
    yield page
    # Any uncaught JS exception fails the test that triggered it.
    assert not errors, f"uncaught JS errors: {errors}"
    page.close()


def test_page_loads_with_expected_controls(page):
    assert "Indian Standards" in page.title()
    for selector in ("#mode-single", "#mode-batch", "#mode-lint",
                     "#query-text", "#submit-btn", "#sample-btn", "#upload-btn"):
        assert page.is_visible(selector) or selector == "#file-input", f"missing {selector}"


def test_single_query_renders_result_cards(page):
    page.fill("#query-text", "structural steel sections for building construction")
    page.click("#submit-btn")
    page.wait_for_selector(".card", timeout=60000)

    cards = page.query_selector_all(".card")
    assert len(cards) >= 1
    assert "IS 2062" in page.inner_text("#results")
    assert page.query_selector(".badge") is not None


def test_explain_button_fetches_and_renders_explanation(page):
    page.fill("#query-text", "structural steel sections")
    page.click("#submit-btn")
    page.wait_for_selector(".explain-btn", timeout=60000)

    page.click(".explain-btn")
    page.wait_for_selector(".explanation", timeout=60000)
    assert "stub explanation" in page.inner_text(".explanation")


def test_batch_mode_renders_per_item_headings(page):
    page.click("#mode-batch")
    page.fill("#query-text", "1. Structural steel sections\n2. Gold jewellery marking")
    page.click("#submit-btn")
    page.wait_for_selector(".batch-item-heading", timeout=60000)

    headings = page.query_selector_all(".batch-item-heading")
    assert len(headings) == 2
    # inner_text reflects CSS text-transform, which uppercases these headings.
    assert "ITEM 1" in headings[0].inner_text().upper()


def test_sample_draft_audit_renders_findings_with_highlight(page):
    """The demo path: load the defective sample, audit it, see findings with
    the offending citation highlighted in context."""
    page.click("#sample-btn")
    assert "IS 2062:2006" in page.input_value("#query-text")

    page.click("#submit-btn")
    # Wait for rendering to settle rather than for the first card, otherwise
    # the text can be read mid-render.
    page.wait_for_function(
        "document.querySelectorAll('.finding').length >= 6", timeout=60000
    )

    results = page.inner_text("#results")
    assert "Superseded citation" in results
    assert "IS 2062:2011" in results          # the suggested fix

    marks = page.query_selector_all(".finding-quote mark")
    assert marks, "no highlighted span rendered"
    assert any("IS 2062:2006" in m.inner_text() for m in marks)


def test_lint_mode_hides_top_k_control(page):
    page.click("#mode-lint")
    assert not page.is_visible("#top-k")
    assert "Audit" in page.inner_text("#submit-btn")

    page.click("#mode-single")
    assert page.is_visible("#top-k")


def test_clean_draft_reports_no_defects(page):
    page.click("#mode-lint")
    page.fill("#query-text",
              "Steel shall conform to IS 2062:2011, tested per IS 1608, "
              "erected per IS 800. All material shall be ISI marked.")
    page.click("#submit-btn")
    page.wait_for_selector(".lint-summary", timeout=60000)
    assert "No defects found" in page.inner_text(".lint-summary")


def test_empty_input_shows_error_without_crashing(page):
    page.click("#submit-btn")
    page.wait_for_selector(".status.error", timeout=10000)
    assert "Enter a" in page.inner_text("#status")


def test_file_upload_audits_document(page, tmp_path):
    draft = tmp_path / "tender.txt"
    draft.write_text("1. Supply of structural steel to IS 2062:2006.\n", encoding="utf-8")

    page.set_input_files("#file-input", str(draft))
    page.wait_for_selector(".finding", timeout=60000)

    assert "Superseded citation" in page.inner_text("#results")
    assert "characters extracted" in page.inner_text("#status")


# -- accessibility -----------------------------------------------------------
#
# IS 17802 (Parts 1 and 2) became legally enforceable for ICT products and
# services on 11 May 2023 via the RPwD Amendment Rules (G.S.R. 359(E)), and
# GIGW 3.0 requires WCAG 2.1 Level AA. For a tool intended for government
# officials this is statutory, not polish — so it is tested, not assumed.

def test_all_interactive_controls_have_accessible_names(page):
    """WCAG 4.1.2 / 3.3.2. A placeholder is not a label: the main textarea and
    the file input were both unnamed, so a screen-reader user could not tell
    what the primary input was for."""
    unnamed = page.eval_on_selector_all(
        "button,input,textarea,select",
        """els => els
            .map(e => ({id: e.id, name: (
                e.getAttribute('aria-label')
                || (e.labels && e.labels.length ? e.labels[0].innerText : '')
                || e.innerText || '').trim()}))
            .filter(c => !c.name)""",
    )
    assert unnamed == [], f"controls without an accessible name: {unnamed}"


def test_results_and_status_are_live_regions(page):
    """WCAG 4.1.3. Findings arrive asynchronously; without a live region the
    entire output of this application is silent to a screen reader."""
    assert page.get_attribute("#status", "aria-live") == "polite"
    assert page.get_attribute("#status", "role") == "status"
    assert page.get_attribute("#results", "aria-live") == "polite"


def test_mode_toggle_exposes_selection_state(page):
    """Three loose buttons cannot convey that the modes are mutually exclusive
    or which one is active. A radiogroup can."""
    assert page.locator("[role=radiogroup]").count() == 1
    assert page.locator("[role=radio]").count() == 3

    page.click("#mode-lint")
    states = page.eval_on_selector_all(
        "[role=radio]", "e=>e.map(x=>[x.id, x.getAttribute('aria-checked')])")
    assert ["mode-lint", "true"] in [list(s) for s in states]
    assert sum(1 for _, checked in states if checked == "true") == 1


def test_heading_structure_and_skip_link(page):
    """WCAG 1.3.1 / 2.4.1 — one h1, section headings under it, and a way past
    the header for keyboard users."""
    tags = page.eval_on_selector_all("h1,h2,h3", "e=>e.map(x=>x.tagName)")
    assert tags[0] == "H1"
    assert tags.count("H1") == 1
    assert "H2" in tags
    assert page.locator(".skip-link").count() == 1


def test_focus_indicator_is_explicit(page):
    """WCAG 2.4.7. The browser default 1px outline is invisible against this
    dark panel background."""
    page.keyboard.press("Tab")
    outline = page.evaluate(
        "()=>{const s=getComputedStyle(document.activeElement);"
        "return {w: s.outlineWidth, c: s.outlineColor}}")
    assert outline["w"] not in ("0px", "medium"), f"no explicit focus outline: {outline}"


def test_results_region_reports_busy_state(page):
    page.click("#mode-lint")
    page.click("#sample-btn")
    page.click("#submit-btn")
    page.wait_for_function("document.querySelectorAll('.finding').length >= 6", timeout=60000)
    assert page.get_attribute("#results", "aria-busy") == "false"
