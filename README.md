# Indian Standards Recommendation & Tender Audit Engine — SIH26108

For Smart India Hackathon 2026, Problem Statement 108 (Bureau of Indian
Standards, Ministry of Consumer Affairs, Food & Public Distribution).

Two modes over a corpus of **6,383 Indian Standards**:

- **Recommend** (`/recommend`, `/batch`) — "which standards apply to this item?"
- **Audit** (`/lint`, `/lint/upload`) — the inverse, and the more valuable one.
  The PS Background describes specs that "omit relevant standards, reference
  outdated versions, or include incomplete technical requirements". Those are
  defects in a draft nobody checks. `/lint` reports them with character spans
  into the officer's own text and a suggested fix.

| Rule | Catches | Fires on |
|---|---|---|
| `superseded_citation` | draft cites IS 2062:2006 → "cite IS 2062:2011" | 671 records (10.5%) |
| `missing_allied_standard` | cites IS 456 but omits IS 383, its normative reference | 901 records (14.1%) |
| `missing_certification_requirement` | product under a QCO, no conformity mark demanded | 18 records (0.3%) |
| `certification_unverified` | says out loud when certification was never checked | always |
| `uncited_item` | line item describes a product but cites no standard | always |
| `unresolved_citation` | cited number matches nothing in the corpus | always |

Every rule is deterministic. **No LLM decides whether a finding fires** —
findings carry procurement and legal consequences, so they must be reproducible
and traceable. The LLM only narrates a finding after the fact.

## Honesty properties

These are design commitments, not features, and the tests enforce them.

- **Silence is never a clearance.** Certification data covers 0.6% of the
  corpus. For the rest the engine reports `certification_status: unknown` with
  `mandatory_certification: null` — never `false` — and the linter emits a
  finding naming the unchecked standards. Reporting "certification not
  required" from data nobody gathered could strip a mandatory ISI requirement
  out of a live tender.
- **Inferred claims are labelled as inferred.** 661 of the 671 supersessions
  are derived from edition years, not read from the BIS catalogue. Those
  findings are reported at `medium`, and say so.
- **Uncertain data is dropped, not guessed.** Annex tables flatten badly in
  scanned text; a reference that cannot be attributed confidently is discarded
  rather than attached to the wrong standard. A fabricated normative reference
  becomes a wrong finding told to an officer.
- **No record claims verification it does not have.** Archive records carry
  `verified: false` and `data_confidence: medium`; a strong match against a
  weakly-sourced record is capped so the UI cannot present it as certain.

## Architecture

Retrieval fuses **four** signals: BM25 (lexical), a local multilingual
embedding model (semantic), exact technical-designation matches (`E250`, `IE3`,
`M25` — near-unique identifiers, so a hit outweighs a soft score), and a
cross-encoder reranker fused as a fourth rank channel rather than overriding
the others.

**Retrieval never leaves the machine.** Procurement text can be sensitive
pre-publication, so the embedding model is self-hosted. The Groq explanation
layer only ever receives already-matched public standard metadata and numeric
scores — never the query — and falls back to a deterministic template.

Superseded editions are demoted, not filtered: the linter still has to
recognise one when a draft cites it.

## Retrieval benchmark

```bash
python3 -m eval.run_eval --gold eval/heldout_queries_v2.jsonl --rerank
```

| set | R@1 | R@5 | MRR |
|---|---|---|---|
| in-lexicon (`gold_queries_v2`, 51 q) | 0.65 | 0.88 | 0.738 |
| **held-out (`heldout_queries_v2`, 22 q)** | **0.27** | **0.73** | **0.466** |

**Quote the held-out number.** Its queries deliberately avoid every alias
string in the lexicon, so it measures generalisation rather than lexicon
recall.

Two things worth knowing about how these numbers moved:

- An earlier version measured **MRR 0.841 held-out on a 39-standard corpus**.
  That did not survive 6,383 candidates. Roughly half the drop was real
  difficulty; the other half was a single-label gold set scoring correct
  retrievals as misses, which is why both sets were relabelled multi-label. A
  small corpus flatters a retrieval metric.
- Undamped chunk max-pooling and an overriding (rather than fused) reranker
  both measurably regressed results. The current constants exist because of
  those measurements, not on principle.

## Running it

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set GROQ_API_KEY for live explanations
uvicorn app.main:app --port 8008
```

`http://127.0.0.1:8008/ui/` — "Load sample draft" runs the audit against a
deliberately defective spec. First boot encodes 6,383 documents (~50s); after
that a content-hashed embedding cache makes startup **0.2s**.

Container: `docker build -f backend/Dockerfile .` — bakes in the model and
embeddings so the first request is fast, runs non-root, and exposes
`/health/live` and `/health/ready` separately (a liveness probe coupled to a
50s index build restarts the pod forever).

## Data pipeline

```bash
python3 -m data_pipeline.build --catalogue          # survey the archive
python3 -m data_pipeline.merge_full_corpus --write  # merge archive + curated
python3 -m data_pipeline.derive_supersession --write
python3 -m data_pipeline.build_allied_graph --write
```

Corpus provenance: 6,264 records with scope extracted from published documents
via the archive.org `gov.in.is.*` collection, plus curated records supplying
the certification, alias and designation layer the rules reason over. Merges
preserve curation; a failed extraction can never blank a good record.

## Tests

```bash
cd backend && python3 -m pytest tests/ -q      # 198 tests, no network needed
playwright install chromium                     # for the browser tests
```

Includes 9 real headless-Chromium tests. Several regression tests exist because
the bug actually happened:

- `VersionResolver.resolve()` consumed a standard's own `superseded_by` list
  with `pop()` — the first request after startup got the right successor and
  **every later one silently reported none**. Only the browser tests caught it,
  being the first to reuse one long-running server the way production does.
- The supersession guard was a tie-break, which fires only on exact score
  equality. It worked at 39 standards and silently stopped at 6,383.
- Non-Latin-script queries were corrupted by BM25 tie-break ordering.

## Known limitations

- **Certification coverage is 0.6% and cannot be derived.** Legal obligations
  do not follow from a scope clause; this needs real QCO notification data.
  The engine is explicit about not knowing, which is the best available
  answer, not a good one.
- **Held-out R@1 is 0.27.** The dominant failure is surface-token matching:
  "protective headgear for labourers" returns *Protective Rubber Canvas Boots*.
  Head-noun matching is the clearest next improvement.
- Aliases cover 39 records. Hand-authoring does not reach 6,383; this needs a
  source such as GeM category mappings.
- One record retains mojibake, mixing damaged and correct text in a way no
  whole-string repair fixes without breaking the good half.
- 3,009 extracted references point outside the corpus and are reported rather
  than resolved.
- `services/knowledge-reasoning/` needs Python 3.11 and is not wired in.
- The React app in `frontend/` has 4 live screens; the other 13 are fixture-
  backed and labelled as such in their own UI.
