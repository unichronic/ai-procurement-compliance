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

Baseline (metadata-only retrieval, no aliases) vs current:

| query_type | R@1 baseline | R@1 now | MRR baseline | MRR now |
|---|---|---|---|---|
| trade_name | 0.64 | **0.93** | 0.738 | 0.964 |
| code_mixed | 0.33 | **1.00** | 0.514 | 1.000 |
| formal_title | 0.80 | **0.90** | 0.900 | 0.950 |
| parametric | 0.86 | **1.00** | 0.929 | 1.000 |
| hindi | 0.88 | 0.75 | 0.938 | 0.844 |
| tender_line | 1.00 | 0.83 | 1.000 | 0.917 |
| **OVERALL** | 0.75 | **0.90** | 0.832 | **0.946** |

Queries missed entirely (absent from top 5): 3 → 0.

### The number that actually matters

The set above shares its vocabulary with the alias lexicon — same author, same
sitting — so it measures *lexicon recall*, not generalisation. There is a
second set, `eval/heldout_queries.jsonl`, written to deliberately avoid all 154
alias strings ("head protection gear for scooter riders" rather than "helmet",
"rolled girders and joists" rather than "MS plate"):

```bash
python3 -m eval.run_eval --heldout
```

| set | R@1 | R@5 | MRR |
|---|---|---|---|
| in-lexicon (`gold_queries`) | 0.90 | 1.00 | 0.946 |
| **held-out (unseen vocabulary)** | **0.64** | 0.95 | **0.759** |

That gap is the honest characterisation of the system: the lexicon works well
for vocabulary it has seen, generalisation is substantially weaker, and the
right standard still lands in the top 5 about 95% of the time either way. Quote
the held-out number, not the other one.

Two further notes on honesty:

- Ingesting real BIS text *lowered* the in-lexicon score (0.96 → 0.90). The
  earlier figure was measured against hand-written scope text that happened to
  be phrased like the hand-written queries. Real documents are less
  accommodating, and that drop is itself evidence the earlier number was
  inflated.
- Undamped chunk max-pooling measurably regressed R@1 (0.90 → 0.86), because
  only part of the corpus has ingested full text and chunk-rich records got
  more chances at a high score. `CHUNK_SCORE_DAMPING` in `retrieval.py` exists
  because of that measurement, not on principle.

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

## Data pipeline

`data_pipeline/` ingests real standards from the public BIS mirror
(law.resource.org → archive.org), which carries full text as well as scans.

```bash
python3 -m data_pipeline.build --catalogue --sections S01 S03 S05   # survey
python3 -m data_pipeline.build --refresh-existing --write           # ingest
python3 -m data_pipeline.build --items gov.in.is.1786.2008 --write  # specific
```

Three sectional indexes alone list **5,082** standards, so the full 14 divisions
are consistent with the ~20,000+ the problem statement describes.

Ingestion extracts the title, ICS code, sectional committee, scope clause,
normative references and body passages from the document itself, so the corpus
is verified against primary sources rather than hand-asserted. Two properties
worth knowing:

- **Curation is never clobbered.** QCO mappings, aliases, supersession chains
  and allied edges survive re-ingest; only document-derived fields refresh. A
  failed extraction can't blank a good record.
- **Uncertain data is dropped, not guessed.** Annex reference tables flatten
  badly in scanned text, and a part-list that can't be confidently attributed
  to its base number is discarded rather than attached to the wrong standard.
  A fabricated normative reference becomes a wrong lint finding, which is worse
  than a missing one. Junk titles ("~", "( Reaffirmed 2002 )") are rejected on
  the same principle.

Each run diffs against `snapshot.json` and reports added/changed/removed
records, because BIS publishes revisions and QCOs continuously and a corpus
that's correct today is wrong in six months.

## Tests

```bash
cd backend
python3 -m pytest tests/ -q
```

157 tests, no network access or API key required. Covers retrieval, citation
parsing across 10 real-world formats, the lint rule engine, the ingestion
pipeline (offline, against fixtures resembling real scanned text), PDF/DOCX
extraction (against genuinely generated files, not mocks), rate limiting,
certification, versioning, allied standards, the explanation layer (Groq
mocked), and the full API surface.

`tests/test_browser_e2e.py` drives headless Chromium against a live server and
clicks through all three UI modes. It self-skips if Playwright's browser isn't
installed:

```bash
playwright install chromium
```

Three regression tests exist because these bugs actually happened:

- non-Latin-script queries getting corrupted by BM25 tie-break ordering, so a
  Hindi query returned gold and LPG standards instead of steel;
- superseded editions outranking active ones;
- `VersionResolver.resolve()` walking a standard's own `superseded_by` list
  with `pop()`, consuming it — the first request after startup got the correct
  successor and **every later one silently reported none.** Only the browser
  tests caught this, because they were the first tests to reuse one
  long-running server across requests, the way production does.

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

- **The corpus is still 39 standards.** The pipeline can reach thousands, but
  only 12 of the 39 resolved against the mirror (it skews to older editions;
  2015+ editions largely aren't there), and nothing has been bulk-ingested yet.
  A linter over 39 standards only catches defects about those 39 — this remains
  the binding constraint on everything else.
- **Generalisation to unseen vocabulary is the weak point**: held-out R@1 is
  0.64 against 0.90 in-lexicon. Growing the alias lexicon by hand has obvious
  limits; mining aliases from GeM catalogue categories or HSN codes is the
  scalable version and isn't built.
- Allied-standard edges are still hand-authored. The pipeline now extracts real
  `referred_standards`, but they aren't yet wired into the allied graph, so the
  linter's normative-reference rule still runs on curated edges.
- Scanned PDFs need OCR before `/lint/upload` can read them; there's no OCR
  step.
- No authentication. Rate limiting is a per-process in-memory limiter, which is
  not a substitute for a gateway, and CORS defaults to localhost (override with
  `ALLOWED_ORIGINS`).
- Web UI is a vanilla HTML/JS page, not integrated into any actual procurement
  portal (GeM/CPPP/IREPS). The REST API is the integration primitive, but no
  portal adapter exists.
