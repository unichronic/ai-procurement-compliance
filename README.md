# Indian Standards Recommendation Engine — SIH26108

Prototype for Smart India Hackathon 2026, Problem Statement 108 (Bureau of
Indian Standards, Ministry of Consumer Affairs, Food & Public Distribution):
an AI-powered engine that turns a procurement officer's product description,
technical spec, or tender document into ranked, applicable Indian Standards
with allied standards, version/amendment status, and certification
requirements.

## Two modes

- **Recommend** (`/recommend`, `/batch`) — "which standards apply to this item?"
- **Audit** (`/lint`) — the inverse, and the more valuable one. The PS Background
  describes specs that "omit relevant standards, reference outdated versions, or
  include incomplete technical requirements." Those are defects in a draft nobody
  is currently checking. `/lint` takes a whole draft spec and reports them, with
  character spans pointing at the offending text and a suggested fix:

  | Rule | Catches |
  |---|---|
  | `superseded_citation` | draft cites IS 2062:2006 → "cite IS 2062:2011" |
  | `missing_allied_standard` | cites IS 2062 but omits IS 1608, its test method |
  | `missing_certification_requirement` | product under a QCO, but the draft never demands the ISI/CRS/hallmark |
  | `uncited_item` | line item describes a product but cites no standard at all |
  | `unresolved_citation` | cited number matches nothing in the corpus |

  Every rule is deterministic and data-driven. **No LLM decides whether a finding
  fires** — findings carry procurement and legal consequences, so they must be
  reproducible and traceable. The LLM only narrates a finding after the fact.

## Architecture

- **Retrieval is entirely local.** A self-hosted multilingual embedding
  model (`paraphrase-multilingual-MiniLM-L12-v2`) combined with BM25 via
  Reciprocal Rank Fusion does the actual matching against procurement text.
  Nothing you type into `/recommend` or `/batch` is sent to any external
  API — this is a deliberate data-sovereignty choice, since tender text can
  be sensitive pre-publication. It also gives free multilingual support:
  Hindi/regional-language queries embed into the same space as English.
- **Explanations use the Groq API**, but only ever receive already-matched,
  already-public standard metadata and numeric match scores — never your
  query text. Falls back to a deterministic template if no `GROQ_API_KEY`
  is set or the API call fails.

Retrieval fuses **three** channels, not two: BM25 (lexical), embeddings
(semantic), and a symbolic channel matching technical designations (`E250`,
`IE3`, `M25`, `22K`, `14.2 kg`). Designations are near-unique identifiers, so a
hit there outweighs a soft similarity score rather than being diluted into it.
Standards also carry `aliases` — the trade names officers actually write
("MS plate", "hard hat", "peene ka paani") which never appear in a BIS title.

Active editions win ties against superseded ones. Without that, a superseded
edition — which shares its number, title and most of its scope with the current
one — can rank first, which is the exact defect this project exists to catch.

See `backend/app/core/*.py` docstrings for the reasoning behind each module.

## Retrieval benchmark

`python3 -m eval.run_eval` scores 51 labelled queries and reports recall/MRR
**broken down by query type**, because an aggregate number hides the only
interesting fact: queries phrased like a BIS title are easy, and queries phrased
the way procurement officers actually write are not.

Effect of adding aliases + the designation channel + status-aware ranking:

| query_type | R@1 before | R@1 after | MRR before | MRR after |
|---|---|---|---|---|
| trade_name | 0.64 | **0.93** | 0.738 | 0.964 |
| code_mixed | 0.33 | **1.00** | 0.514 | 1.000 |
| formal_title | 0.80 | **0.90** | 0.900 | 0.950 |
| parametric | 0.86 | **1.00** | 0.929 | 1.000 |
| hindi | 0.88 | **1.00** | 0.938 | 1.000 |
| tender_line | 1.00 | 1.00 | 1.000 | 1.000 |
| **OVERALL** | 0.75 | **0.96** | 0.832 | **0.980** |

Queries missed entirely (absent from top 5): 3 → 0.

**Read these numbers with the caveat that they are optimistic.** The aliases and
the gold queries were authored by the same person in the same sitting, so the
lexicon has effectively seen the test set. What the benchmark honestly
demonstrates is that *the mechanism works* — trade-name and code-mixed retrieval
were genuinely broken before and are not now, and adding aliases did not degrade
formal-title retrieval. What it does **not** establish is generalisation to trade
names nobody thought to add. A clean measurement needs held-out queries written
by someone who didn't write the lexicon.

## Running it

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set GROQ_API_KEY if you want live explanations
uvicorn app.main:app --reload --port 8008
```

Open `http://127.0.0.1:8008/ui/` for the web UI ("Load sample draft" shows the
audit mode against a deliberately defective spec), or use the REST API directly
(`/recommend`, `/batch`, `/lint`, `/explain`, `/standard/{id}`, `/standards`,
`/qco-orders`, `/health`).

## Tests

```bash
cd backend
python3 -m pytest tests/ -q
```

92 tests covering retrieval (including regression tests for two real bugs found
during development: non-Latin-script queries corrupted by BM25 tie-break
ordering, and superseded editions outranking active ones), citation parsing
across 10 real-world formats, the lint rule engine, certification, versioning,
allied standards, span-preserving document splitting, the explanation layer
(Groq mocked, so the suite needs no network access or API key), and the full API
surface via FastAPI's `TestClient`.

## Status against the official PS26108 "Expected Features"

| Requirement | Status |
|---|---|
| Accept product descriptions / specs / tender documents as input | Done — `/recommend`, `/batch` |
| Recommend by semantic understanding, not keyword matching | Done — hybrid BM25 + semantic RRF |
| Identify allied standards (normative reference, test method, terminology, safety, installation, related product) | Done |
| Highlight latest version / amendments | Done |
| Suggest mandatory certification (BIS/CRS/Hallmarking) | Done |
| Support multilingual input / natural language queries | Done |

## Known limitations

- `backend/app/data/standards.json` is a curated seed set (39 standards),
  not the full ~20,000+ BIS corpus. `data_pipeline/` (real ingestion from
  BIS) is not yet built. A linter over 39 standards can only catch defects
  about those 39 — this is the binding constraint on everything above.
- Standards are indexed on title/scope/aliases, i.e. **metadata, not document
  content**. Real discrimination between similar standards lives in scope
  clauses and designation tables. The data model accepts a `chunks` field for
  this, but nothing populates it yet.
- The alias lexicon is hand-authored and small; see the benchmark caveat above.
- No file upload — `/lint` and `/batch` take pasted plain text, not PDF/DOCX.
- Web UI is a minimal vanilla HTML/JS page, not integrated into any actual
  procurement portal (GeM/CPPP/IREPS/etc).
- No authentication or rate limiting; CORS is wide open. Local-demo posture.
