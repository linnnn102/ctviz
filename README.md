# ClinicalTrials.gov Query-to-Visualization Agent

A FastAPI backend that turns a clinical-trials question ("Which drugs frequently co-occur in combination studies for pancreatic cancer?") into a **structured visualization spec** backed by live [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) data. Every datum (bar, time bucket, histogram bin, scatter point, network node, network edge) carries **deep citations**: the NCT id plus the exact API field path and verbatim value that put that trial in that datum.

Supported outputs: `bar_chart`, `grouped_bar_chart`, `stacked_bar_chart`, `time_series` (single or multi-series), `histogram`, `scatter_plot`, `network_graph` (co-occurrence or bipartite), `metric`, `table`. A small demo UI is included at `/`.

---

## 1. How to run

Requirements: Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/) (or plain pip, see below).

```bash
uv sync                                     # install
uv run uvicorn app.main:app --port 8077     # start  (use any free port)
open http://127.0.0.1:8077                  # demo UI;  API docs at /docs
```

```bash
curl -s -X POST http://127.0.0.1:8077/visualize \
  -H 'content-type: application/json' \
  -d '{"query": "How has the number of trials for this drug changed over time?", "drug_name": "Pembrolizumab"}'
```

**Configuration.** Everything works with no configuration. To turn on the LLM planner cascade (gpt-5-mini, escalating to gpt-5.4), provide an OpenAI API key:

```bash
cp .env.example .env        # then fill in OPENAI_API_KEY
uv run --env-file .env uvicorn app.main:app --port 8077
```

Without a key (or with `"use_llm": false` in a request) the deterministic rule-based planner is used. `GET /health` reports which planner and models are active. Other settings are listed in `.env.example`.

**Without uv:** `python -m venv .venv && . .venv/bin/activate && pip install fastapi "uvicorn[standard]" httpx pydantic openai pytest pytest-asyncio respx`, then run the same commands without the `uv run` prefix.

**Tests and evaluation**

```bash
uv run pytest              # 84 offline tests (API mocked with respx, LLMs stubbed), ~4 s
uv run pytest -m live      # 5 tests against the real API, including citation verification
uv run --env-file .env python scripts/run_examples.py   # regenerate examples/ from live data
uv run --env-file .env python evals/run_eval.py         # planner model eval (~400 LLM calls, ~$1)
```

### Live deployment

A public instance runs on Render's free tier: **https://ctviz.onrender.com** (UI at `/`, docs at `/docs`). Update this URL if the service name differs.

- **Cold start.** The free instance sleeps after 15 minutes idle, so the first request can take about 30–60 s.
- **Cost guardrails.** Each client IP is limited to 10 requests per minute (`CTVIZ_RATE_LIMIT_PER_MIN`, returns 429). The LLM planner is capped at 200 requests per day (`CTVIZ_DAILY_LLM_LIMIT`); after that the service falls back to the rule-based planner and says so in `meta.warnings`. `GET /health` reports `llm_requests_today`.
- **Deploying your own copy.** [render.yaml](render.yaml) is a Render Blueprint: push the repo to GitHub, choose *New → Blueprint* in Render, and enter `OPENAI_API_KEY` when prompted (leave it empty for rules only). Render builds from [requirements.txt](requirements.txt), which is exported from `uv.lock` with `uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/visualize` | question → visualization spec + metadata (main endpoint) |
| POST | `/plan` | question → validated plan only (debugging / transparency) |
| GET | `/dimensions` | catalog of dimensions the engine can group by |
| GET | `/health` | liveness and planner status |
| GET | `/docs`, `/openapi.json` | generated OpenAPI docs (same Pydantic models as below) |

---

## 2. Architecture

```
               ┌─────────────────────────── planning (LLM cascade) ───────────────────────────┐   ┌──────────── execution (deterministic) ────────────┐
request ──►  gpt-5-mini ──► validator ──ok──────────────────────────────────────────────────►  CT.gov client ──► aggregation ──► spec builder ──► response
                 │ error / low confidence / severe repair                              ▲       paging, retry,     dimension         encodings,
                 ▼                                                                     │       cache, field       registry with     titles, meta
             gpt-5.4 ──► validator ─────────────────────────────────────────────────────┤       projection         citations
                 │ error                                                               │
                 ▼                                                                     │
             gpt-5-mini's validated plan, else rule planner ──► validator ─────────────┘
```

Both models use OpenAI structured outputs (`responses.parse(text_format=LLMPlan)`) and return an enum-constrained `QueryPlan` plus a self-reported `confidence`. Section 5a explains how the models were chosen.

The central design choice is a single small contract, **`QueryPlan`** ([app/plan.py](app/plan.py)): intent, dimensions, filters, cohorts, and network spec, all drawn from closed vocabularies ([app/vocab.py](app/vocab.py)). The planners only *interpret* the question. Everything that produces numbers is deterministic code operating on API records. The LLM never sees trial data and never writes a value that appears in a chart.

| Module | Responsibility |
|---|---|
| [app/schemas.py](app/schemas.py) | Public request/response models (this is the schema documentation; FastAPI renders it at `/docs`) |
| [app/plan.py](app/plan.py), [app/vocab.py](app/vocab.py) | QueryPlan contract and closed vocabularies |
| [app/planner/llm.py](app/planner/llm.py) | OpenAI structured-output call: question → `QueryPlan` + confidence, with token usage and cost |
| [app/planner/\_\_init\_\_.py](app/planner/__init__.py) | The cascade: small model → escalate to gpt-5.4 → rules; per-call trace |
| [evals/](evals/) | Planner model evaluation: 50 queries, scorer, cascade simulation, reports |
| [app/planner/rules.py](app/planner/rules.py) | Keyword/regex planner (fallback, and the default without a key) |
| [app/planner/validate.py](app/planner/validate.py) | Grounding, structured-field overrides, intent/dimension repair; every change logged |
| [app/ctgov/](app/ctgov/) | Filter → API param translation; async client with paging, retries, and a TTL cache |
| [app/dimensions.py](app/dimensions.py) | Registry of 22 dimensions; each extractor returns values *with* their source field path and verbatim text |
| [app/normalize.py](app/normalize.py) | Real-data cleanup: drug names, sponsor suffixes, phase buckets, dates |
| [app/analysis/aggregate.py](app/analysis/aggregate.py) | Pure aggregation functions (distribution, time trend, histogram, scatter, network, table, count) |
| [app/analysis/engine.py](app/analysis/engine.py), [app/viz.py](app/viz.py) | Orchestration and spec building |

**Coverage without one-off hacks.** There are seven intents (`count`, `distribution`, `time_trend`, `histogram`, `scatter`, `network`, `list`) and 22 dimensions, and any combination works. Cohorts turn any of them into a comparison. Adding a dimension means one registry entry in `dimensions.py`. For example, "network of sponsors and conditions for pembrolizumab trials" and "melanoma trials completing per year by status" work with no new code. "Enrollment vs number of sites" restricted to NIH sponsors also works, but the rule planner needs `sponsor_class: ["NIH"]` passed as a structured field; only the LLM planner maps the phrase "NIH-sponsored" to that filter.

---

## 3. Request schema

`POST /visualize`, JSON body. Unknown fields are rejected (`422`).

| Field | Type | Required | Default | Validation / meaning |
|---|---|---|---|---|
| `query` | string | **yes** | — | 3–1000 chars. Natural-language question. |
| `drug_name` | string | no | null | ≤200 chars. Intervention search (`query.intr`, which uses CT.gov synonym expansion, e.g. Keytruda → pembrolizumab). |
| `condition` | string | no | null | ≤200. Condition search (`query.cond`). |
| `sponsor` | string | no | null | ≤200. Sponsor/collaborator search (`query.spons`). |
| `country` | string | no | null | ≤100. Trials with ≥1 site in this country, using CT.gov's name (e.g. `"United States"`, `"Korea, Republic of"`). |
| `trial_phase` | array of enum | no | null | `EARLY_PHASE1, PHASE1, PHASE2, PHASE3, PHASE4, NA` (OR). |
| `status` | array of enum | no | null | `RECRUITING, NOT_YET_RECRUITING, ENROLLING_BY_INVITATION, ACTIVE_NOT_RECRUITING, SUSPENDED, TERMINATED, COMPLETED, WITHDRAWN, UNKNOWN`. |
| `study_type` | enum | no | null | `INTERVENTIONAL, OBSERVATIONAL, EXPANDED_ACCESS`. |
| `intervention_type` | array of enum | no | null | `DRUG, BIOLOGICAL, DEVICE, PROCEDURE, RADIATION, BEHAVIORAL, GENETIC, DIETARY_SUPPLEMENT, COMBINATION_PRODUCT, DIAGNOSTIC_TEST, OTHER`. |
| `sponsor_class` | array of enum | no | null | `INDUSTRY, NIH, FED, OTHER_GOV, INDIV, NETWORK, AMBIG, OTHER, UNKNOWN`. |
| `start_year` / `end_year` | int | no | null | 1900–2100; `start_year ≤ end_year`. Filters on trial **start date**. |
| `compare` | array of string | no | null | 2–6 items. Explicit cohorts, e.g. `["pembrolizumab","nivolumab"]`. |
| `compare_on` | enum | no | `"drug"` | `drug, condition, sponsor, country`: which filter the `compare` values set. |
| `chart_type` | enum | no | null | Forces a chart type. Ignored, with a note, if incompatible with the intent. |
| `group_by` | enum | no | null | Forces the primary dimension (see `GET /dimensions`). |
| `top_n` | int | no | 15 | 3–100. Maximum categories, or nodes per type, shown. |
| `max_studies` | int | no | 5000 | 50–10000. Maximum records analyzed **per cohort**. |
| `max_citations_per_datum` | int | no | 5 | 0–50. |
| `include_nct_ids` | bool | no | false | Adds the full list of contributing NCT ids to every datum. |
| `use_llm` | bool | no | true | `false` forces the rule-based planner. |

**Precedence.** Structured fields always override what the planner extracted from `query`, and each override is recorded in `meta.interpretation.adjustments`.

---

## 4. Response schema

```jsonc
{
  "visualization": {
    "type": "bar_chart | grouped_bar_chart | stacked_bar_chart | time_series | histogram | scatter_plot | network_graph | metric | table",
    "title": "Trials by phase: Pembrolizumab vs Nivolumab",
    "subtitle": "string | null",
    "encoding": { ... },          // which data keys go to which visual channel (below)
    "data": [ ... ] | { "nodes": [ ... ], "edges": [ ... ] },
    "options": {
      "orientation": "vertical | horizontal | null",   // bar charts
      "stacked": false,
      "time_granularity": "year | null",
      "x_scale": "linear | log | null", "y_scale": "...",  // scatter: log when range spans >1000x
      "layout": "force | null", "bipartite": true          // network
    }
  },
  "meta": {
    "query": "...",
    "source": "clinicaltrials.gov", "source_api": "https://clinicaltrials.gov/api/v2/studies",
    "data_as_of": "2026-10-20",                          // CT.gov dataTimestamp
    "filters": { "condition": "melanoma", "statuses": ["RECRUITING"] },
    "interpretation": {
      "planner": "llm | rules",
      "model": "gpt-5-mini",       // LLM whose plan was executed; null for the rule planner
      "escalated": false,          // true if gpt-5-mini was overruled by gpt-5.4
      "escalation_reason": null,   // e.g. "low confidence (...)" or "validator had to repair the plan: ..."
      "intent": "distribution",
      "rationale": "...",
      "plan": { ... },             // the exact validated QueryPlan that was executed
      "adjustments": [ "dropped ungrounded global filter sponsor='Pfizer' ..." ],
      "llm_trace": [               // one entry per LLM call, in order
        { "model": "gpt-5-mini", "outcome": "accepted | escalated | failed", "latency_ms": 5210,
          "input_tokens": 2718, "output_tokens": 512, "reasoning_tokens": 256,
          "confidence": "high | medium | low", "cost_usd": 0.00106, "detail": "why escalated / failed" }
      ]
    },
    "coverage": [ { "label": "Melanoma", "total_matching": 3778, "analyzed": 3778, "truncated": false,
                    "api_params": { "query.cond": "melanoma" } } ],
    "counting_rule": "Each trial is counted once per distinct country it lists ...",
    "sort": "trial_count descending (encoding.x.sort)",
    "units": { "trial_count": "trials", "enrollment": "participants" },
    "warnings": [ ... ],   // things that affect correctness: truncation, zero results, fallbacks
    "notes": [ ... ],      // informational: top-N cut, excluded missing values, stats
    "generated_at": "2026-10-20T18:01:02+00:00", "elapsed_ms": 2303
  }
}
```

### Encoding

Each channel is a `FieldEncoding`: `{field, type: nominal|ordinal|quantitative|temporal, title, unit, sort, aggregate}`. `field` is the key to read from each data row. `sort`, when it is a list, is the **exact category order** to render. It is always given for categorical axes, and the renderer should not re-sort.

| `type` | Channels used | Row shape |
|---|---|---|
| `bar_chart` | `x` (category), `y` (`trial_count`) | `{<dim>: "Phase 3", trial_count, percent, citations, citation_count}` |
| `grouped_bar_chart` / `stacked_bar_chart` | `x`, `y`, `series` (`cohort` or a dimension) | one row per (category, series), **zeros included** |
| `time_series` | `x` (`year`, temporal), `y`, optional `series` | one row per year per series; missing years are filled with `trial_count: 0` |
| `histogram` | `x` (`bin_start`), `x2` (`bin_end`), `y`, `label` (`bin_label`) | `{bin_start, bin_end, bin_label, overflow, trial_count, ...}`; bins are `[start, end)`; the last bin may be an overflow bin (`overflow: true`) |
| `scatter_plot` | `x`, `y` (numeric dims), optional `series`, `label` (`title`) | one row per trial: `{nct_id, title, <x>, <y>, citations}` |
| `network_graph` | `network: {node_id, node_label, node_type, node_size, edge_source, edge_target, edge_weight}` | `data.nodes[]`: `{id, label, type, trial_count, degree, is_query_entity, citations}`; `data.edges[]`: `{source, target, weight, citations}` |
| `metric` | `value` (`trial_count`), `label` | one row: `{cohort, trial_count, citations}`; the count is the API's exact `totalCount` |
| `table` | `columns[]` | one row per trial (or per edge, if a network was forced to `table`) |

`percent` is the share of *analyzed* trials in that cohort or series, so grouped bars from cohorts of different sizes stay comparable.

### Citations (on every row, node and edge)

```json
"trial_count": 41,
"citation_count": 41,
"citations": [
  {
    "nct_id": "NCT03304639",
    "url": "https://clinicaltrials.gov/study/NCT03304639",
    "title": "Pembrolizumab With or Without Stereotactic Body Radiation Therapy ...",
    "excerpts": [
      {"field": "protocolSection.armsInterventionsModule.interventions[0].name", "text": "Pembrolizumab"},
      {"field": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name", "text": "National Cancer Institute (NCI)"}
    ]
  }
]
```

`excerpts[].field` is a resolvable path into the study JSON returned by `GET /api/v2/studies/{nct_id}`, and `text` is the verbatim value at that path. Network edges cite **both** endpoints from the same trial. `citations` holds at most `max_citations_per_datum` entries; `citation_count` says how many trials could be cited, and `include_nct_ids: true` returns all of them.

The full set of Pydantic models is in [app/schemas.py](app/schemas.py) and `/openapi.json`.

---

## 5. Example runs

Real outputs from the live API, saved as request/response pairs. They were generated with `max_citations_per_datum: 2` to keep the files readable.

| File | Question | Output |
|---|---|---|
| [examples/01_time_trend_pembrolizumab.json](examples/01_time_trend_pembrolizumab.json) | "How has the number of trials for this drug changed over time?" + `drug_name: Pembrolizumab` | `time_series`, 2,987 trials, all analyzed |
| [examples/02_compare_phases_pembro_vs_nivo.json](examples/02_compare_phases_pembro_vs_nivo.json) | "Compare phases for trials involving Pembrolizumab vs Nivolumab" | `grouped_bar_chart`, two cohorts |
| [examples/03_countries_recruiting_lung_cancer.json](examples/03_countries_recruiting_lung_cancer.json) | "Which countries have the most recruiting trials for lung cancer?" | horizontal `bar_chart` |
| [examples/04_network_sponsor_drug_melanoma.json](examples/04_network_sponsor_drug_melanoma.json) | "Show a network of sponsors ↔ drugs for melanoma trials" | bipartite `network_graph` (e.g. BMS–ipilimumab 48, Merck–pembrolizumab 27) |
| [examples/05_network_drug_cooccurrence_pancreatic.json](examples/05_network_drug_cooccurrence_pancreatic.json) | "Which drugs frequently co-occur in combination studies for pancreatic cancer?" | co-occurrence `network_graph` (gemcitabine–nab-paclitaxel is the strongest edge; the FOLFIRINOX components cluster) |

More chart types are in [examples/extra/](examples/extra/): a histogram (enrollment, phase 3 diabetes), a scatter plot (enrollment vs duration, Alzheimer's), grouped bars (sponsor class, asthma vs COPD, via `compare`), a multi-series time trend (melanoma by phase since 2015), a metric (recruiting semaglutide trials), and a table (recruiting glioblastoma trials in Germany).

> All saved examples went through the LLM cascade. gpt-5-mini answered every one without escalation, at about $0.001 per query. `meta.interpretation.llm_trace` records the model, tokens, latency and cost of each call.

## 5a. Choosing the planner model (accuracy vs cost)

**Setup.** The cascade needs a cheap first-tier model. Three candidates (gpt-5-mini, gpt-5.4-nano, gpt-4.1-mini) were evaluated on 50 queries ([evals/queries.json](evals/queries.json)) covering every intent, comparisons, networks, structured fields, typos, relative dates and an off-topic question.
- **Reference.** gpt-5.4's validated plans are the reference answers.
- **Scoring.** Each candidate's executed plan is compared on seven components: intent, dimensions, series, entity filters, enum filters, years and cohorts. *Exact* means all seven match.
- **Repeats.** Each candidate ran twice.
- **Ceiling.** A second, independent gpt-5.4 pass measures gpt-5.4's own run-to-run noise. The same pass supplies the answers for escalated queries, so an escalation isn't automatically counted as correct.
- **Selection rule**, fixed before the results: among candidates within 3 points of the best cascaded accuracy, pick the cheapest per request.

Results from [evals/results/20261021-1358/report.md](evals/results/20261021-1358/report.md):

| Model | Exact (solo) | Escalation rate | **Exact (cascade)** | $/request solo | **$/request cascade** | p50 latency |
|---|---|---|---|---|---|---|
| gpt-5.4 (2nd pass = ceiling) | 98% | — | — | $0.00789 | — | 4.2 s |
| **gpt-5-mini** ✅ | 93% (92–94) | 7% | **94%** | $0.00106 | **$0.00170** | 5.2 s |
| gpt-4.1-mini | 86% (84–88) | 13% | 88% | $0.00078 | $0.00205 | 1.8 s |
| gpt-5.4-nano | 78% (74–82) | 17% | 86% | $0.00058 | $0.00212 | 2.3 s |

*Exact (solo)* is the mean of the two runs, with the min–max range in parentheses. *Escalation rate* is the share of queries the cascade would send to gpt-5.4. All runs had 0 API errors.

**Accuracy by plan component** (solo, mean of runs). This shows *where* each model goes wrong:

| Model | intent | dimensions | series_by | entity filters | enum filters | years | cohorts |
|---|---|---|---|---|---|---|---|
| gpt-5.4 (2nd pass) | 100% | 100% | 100% | 98% | 100% | 100% | 100% |
| gpt-5-mini | 100% | 98% | 100% | 98% | 99% | 98% | 100% |
| gpt-4.1-mini | 98% | 98% | 100% | 95% | 98% | 97% | 100% |
| gpt-5.4-nano | 98% | 93% | 100% | 88% | 98% | 98% | 98% |

**Exact-match accuracy by question category** (solo, pooled over both runs). With 2–10 queries per category, read these as indicative only:

| Category | Queries | gpt-5.4 (2nd pass) | gpt-5-mini | gpt-4.1-mini | gpt-5.4-nano |
|---|---|---|---|---|---|
| time_trend | 7 | 86% | 79% | 64% | 64% |
| distribution | 10 | 100% | 100% | 100% | 80% |
| comparison | 7 | 100% | 100% | 100% | 86% |
| geographic | 4 | 100% | 88% | 75% | 75% |
| network | 7 | 100% | 93% | 86% | 86% |
| histogram | 3 | 100% | 83% | 100% | 50% |
| scatter | 2 | 100% | 75% | 100% | 100% |
| count | 2 | 100% | 100% | 100% | 25% |
| list | 2 | 100% | 100% | 100% | 75% |
| structured fields | 2 | 100% | 100% | 100% | 100% |
| robustness (typos, mixed filters, off-topic) | 4 | 100% | 100% | 38% | 100% |

Time trends are hardest for every model, gpt-5.4 included. The ambiguous cases are "registered each year" (start year or first-posted year?) and "last 5 years" (off-by-one, and whether to cap at the current year). gpt-4.1-mini's robustness misses come from three cases. It kept a typo ("pembrolizumb") in both runs, put "device" in a text filter instead of the intervention-type filter, and picked a different intent from the reference for the off-topic weather question (a harmless miss).

**How the numbers changed across iterations.** Each row is either a new run or a rescore of saved outputs with fixed code. A rescore re-runs the validator and scorer on the same model outputs, so it costs nothing and keeps the comparison fair. Exact (solo) / exact (cascade):

| Step | gpt-5-mini | gpt-4.1-mini | gpt-5.4-nano | gpt-5.4 ceiling |
|---|---|---|---|---|
| Run 1 (initial prompt, validator and scorer) | 82% / 89% | 75% / 84% | 72% / 82% | 96% |
| Run 1 rescored: grounding bug and scorer strictness fixed | 89% / 92% | 84% / 86% | 74% / 83% | 96% |
| Run 2: prompt forbids unrequested type filters | 92% / 93% | 85% / 87% | 77% / 85% | 98% |
| Run 2 rescored: validator removes design-only filters (**final**) | **93% / 94%** | 86% / 88% | 78% / 86% | 98% |

The ranking was the same at every step. Only the final run's raw outputs are kept, in [evals/results/20261021-1358/](evals/results/20261021-1358/): `raw.jsonl` holds every model output, `summary.json` the metrics, and `report.md` the per-query disagreements.

**Cost inputs** (USD per 1M tokens, OpenAI list prices, October 2026; set in [app/config.py](app/config.py)):

| Model | Input | Cached input | Output |
|---|---|---|---|
| gpt-5.4 | 2.50 | 0.25 | 15.00 |
| gpt-5-mini | 0.25 | 0.025 | 2.00 |
| gpt-4.1-mini | 0.40 | 0.10 | 1.60 |
| gpt-5.4-nano | 0.20 | 0.02 | 1.25 |

A planning call uses about 2.7k input tokens (mostly the fixed instructions) and 300–800 output tokens, reasoning included. One full eval run (about 400 calls) costs roughly $1.

**Decision: gpt-5-mini, escalating to gpt-5.4.**
- It is both the most accurate and the cheapest cascade. The cheaper models escalate so often that their gpt-5.4 calls cost more than the per-call savings.
- The cascade costs about 78% less than calling gpt-5.4 every time, while staying 4 points below gpt-5.4's own run-to-run ceiling.
- The tradeoff is latency: gpt-4.1-mini is roughly 3× faster for 6 points less accuracy. It is the better choice if latency matters more than accuracy (`CTVIZ_PLANNER_MODEL=gpt-4.1-mini`).

**What the eval caught (and what was fixed).**
- **Grounding bug.** The first run showed that the validator was dropping correct but fuller entity names, such as "Parkinson's disease" for a question about "Parkinson's trials". In production that would silently widen the query to all trials. Grounding now works token by token and tolerates generic words like "disease", with regression tests.
- **Scorer strictness.** The scorer was also too strict: a shared filter placed inside each cohort was counted as a miss even though the queries are identical. It now compares each cohort's effective filters.
- **Prompt gap.** All models tended to add unrequested `intervention_types` filters to network questions, so the prompt rule was tightened and the full eval re-run. The validator now also removes filters that only restate the question's structure: a `term` like "combination", or a drug-type filter on a drug-dimension chart. The numbers above are from that final run, rescored with the final validator.

**Caveats.**
- Because gpt-5.4 is the reference, cases where gpt-5.4 is wrong count against the candidates. For example, "network of conditions studied together with pembrolizumab": gpt-5.4 chose condition↔drug, while the candidates chose condition↔condition, which is arguably better.
- 50 queries means each query is worth 2 points, so small differences are within noise.
- Prices are hard-coded in [app/config.py](app/config.py) from OpenAI's published rates as of October 2026.

---

## 6. Key design decisions and tradeoffs

1. **The LLM plans; code computes.** The model fills an enum-constrained `QueryPlan` through OpenAI structured outputs (`client.responses.parse(text_format=LLMPlan)`, strict JSON schema). It cannot invent dimensions, chart types, phases or statuses, and it never handles data. This removes the main hallucination risk: made-up numbers.
2. **Validation after planning, whichever planner ran** ([validate.py](app/planner/validate.py)):
   - *Grounding.* Every free-text entity (drug, condition, sponsor, country, term) must appear in the question or the structured fields. Fuzzy matching tolerates typos, and country aliases are accepted ("US" → United States). Anything else is dropped and reported.
   - *Repair.* Incompatible combinations are fixed with documented defaults instead of failing. Examples: a temporal `group_by` turns a distribution into a time trend; a network over non-entity dimensions becomes drug ↔ drug; a single cohort is merged into the global filters.
   - *Transparency.* The executed plan and every adjustment are returned in `meta.interpretation`.
3. **Graceful degradation.** If the LLM is unavailable, rate-limited, refuses, truncates, or returns unparseable output, the request falls back to the rule planner with a warning. If an entity-specific search returns 0 trials, it is retried once as a free-text search, with a warning.
4. **Use the API's own search semantics.** Entities go into `query.intr` / `query.cond` / `query.spons`, which inherit CT.gov's synonym expansion. Enumerated filters use `filter.advanced` Essie expressions (`AREA[Phase](PHASE2 OR PHASE3)`, `AREA[StartDate]RANGE[...]`, `AREA[LocationCountry]"..."`). All of these were verified against the live API before being used.
5. **Fetch, then aggregate locally**, with `fields=` projection and pages of 1,000 records. One code path serves every chart type, and per-trial citations come for free. The cost is that large result sets are capped by `max_studies` (default 5,000, maximum 10,000); `meta.coverage` and `warnings` always say when that happens. For pure counts (`metric`), the exact `totalCount` is used and only a few records are fetched for citations.
6. **Real-world data handling** ([normalize.py](app/normalize.py)):
   - *Drug names.* Folded in this order: the searched drug's synonyms (via `otherNames`, so "MK-3475" counts as pembrolizumab), then the trial's own MeSH intervention terms, then regex cleanup of doses and dosage forms ("Pembrolizumab 100 mg/4 mL Injection" → pembrolizumab). Hyphens count as part of a word, so nab-paclitaxel stays distinct from paclitaxel. Placebo is excluded from the drug dimension.
   - *Sponsors.* Grouped after stripping corporate suffixes ("Merck Sharp & Dohme LLC" and "… Corp." are one sponsor); the most common raw spelling is displayed.
   - *Conditions.* Use MeSH terms when available, otherwise the raw condition text.
   - *Phases.* `["PHASE1","PHASE2"]` becomes the ordinal bucket "Phase 1/Phase 2"; a missing phase becomes "Unknown".
   - *Missing values.* Counted and reported in `notes`, never silently dropped.
7. **Explicit counting semantics.** For multi-valued dimensions (country, drug, condition) a trial is counted once per distinct value, so bar totals can exceed the trial count. `meta.counting_rule` says this in plain words, and `percent` is provided for comparisons.
8. **Frontend-oriented output.** Category order is fixed explicitly, grouped bars include explicit zero rows, time series have gap years filled, bins carry ready-made labels, and network node ids are typed (`drug:pembrolizumab`) so a drug and a sponsor with the same name can't collide.
9. **Network pruning.** Top-N nodes per type, counting only nodes that have at least one edge, then the strongest `6×top_n` edges, with isolated nodes removed. The remaining structure is reported in `notes`. This keeps the graph readable while being honest about what was cut.

---

## 7. Limitations and what I'd improve with more time

- **Truncation can bias distributions.** When more than `max_studies` trials match, the first N records in API order are analyzed. Better options: per-category `countTotal` queries for exact counts of single-valued enumerated dimensions (phase, status, sponsor class), or random page sampling.
- **The rule planner is heuristic.** It handles the documented question classes but can misread unusual phrasing. For example, "compare sponsor categories across two conditions" without naming the conditions needs the `compare` field. The LLM planner covers that breadth. The next step would be a small evaluation set of (question → expected plan) pairs to score both planners.
- **Escalation relies partly on self-reported confidence.** A small model that is confidently wrong in a way the validator can't detect (for example, a valid but wrong dimension) is not escalated. Better: compare the small model's plan with a second cheap sample, and escalate when they disagree.
- **The eval reference is gpt-5.4, not human labels.** A hand-reviewed gold set would also measure gpt-5.4's own errors.
- **Planner latency.** gpt-5-mini adds about 5 s per request. Prompt caching (the instructions are a stable prefix) and lower reasoning effort are untested levers.
- **Entity resolution is string-based.** Drug normalization covers dose forms, MeSH matches and the searched drug's synonyms, but not, for example, brand ↔ generic names for drugs that weren't searched. A proper fix would map to RxNorm/MeSH identifiers.
- **Country is trial-level.** "Recruiting trials in X" uses the trial's overall status, not per-site recruitment status (`locations[].status`).
- **Time granularity is year only.** Month and quarter would be straightforward additions in `time_trend`.
- **Single process.** The cache is in memory and there is no API-side rate limiting. Production would need Redis caching, request queuing against CT.gov's per-IP limit, auth, and structured logging/metrics.
- **No streaming or progress reporting.** End-to-end latency for the saved examples is 4–10 s (median about 7 s). Roughly 5 s of that is the gpt-5-mini planning call; the rest is ClinicalTrials.gov fetching and aggregation.

---

## 8. Tools, validation, and what was deliberate vs generated

**Tools used.** This project was built with Claude Code (Anthropic's coding agent), working from the assignment brief.
- *Decided by me:*
  - the stack (Python + FastAPI), a rule-based fallback so the service runs without an API key, and a small demo UI;
  - OpenAI models, with gpt-5.4 for semantic understanding delegating to a smaller model;
  - the candidate models to test, selection by accuracy and cost, gpt-5.4 as the eval reference, and the cascade routing.
- *Proposed and implemented by Claude Code:* the architecture (the plan/execute split around `QueryPlan`, the grounding validator, the dimension registry with built-in citations), the escalation criteria, the eval harness and scoring, the code, the tests, and this README.

All code was reviewed to make sure the fallback strategy and the guardrails work.

Libraries: FastAPI, Pydantic v2, httpx, the OpenAI Python SDK, pytest/respx, and ECharts for the demo UI.

**How correctness was validated.**
- *API behavior before relying on it.* Before writing the client, every filter (phase OR-lists, date ranges, country, sponsor class, intervention type, status lists) was checked live against `totalCount`.
- *Offline tests* (`uv run pytest`, 84 tests): normalization edge cases from real data; exact counts and citations on fixture studies shaped like the API projection; time-trend zero filling; histogram bins partitioning trials exactly; co-occurrence and bipartite networks; overlapping cohorts not double-counted; validator grounding and repair; HTTP 422/502 handling; pagination, retry and truncation; zero-result relaxation; every cascade path with stubbed models (accept, escalate on ungrounded entity, escalate on low confidence, keep the small plan if gpt-5.4 fails, rules if both fail); the eval scorer's equivalence rules.
- *Live tests* (`uv run pytest -m live`): for several queries, cited trials are **re-fetched independently** from `/studies/{nct_id}`, and every `excerpt.field` is resolved and compared to `excerpt.text`. All match verbatim. A second test checks that an untruncated single-valued bar equals the API's own filtered count exactly.
- *Face validity.* Network outputs were checked against known oncology practice: gemcitabine + nab-paclitaxel is the top pancreatic combination, FOLFIRINOX drugs cluster, and BMS links to ipilimumab/nivolumab while Merck links to pembrolizumab.

**Iteration driven by real data.** Testing against live data exposed several bugs that were then fixed:
- "Pembrolizumab 100 mg/4 mL" had left a dangling "ml" token after dose stripping.
- MeSH matching folded nab-paclitaxel into paclitaxel.
- "combi**nation**" matched the country keyword.
- "phase 3" was read as a grouping rather than a filter.
- Scatter questions ("enrollment vs duration") were mistaken for cohort comparisons.
- Bursts of single-study requests hit the API's rate limit, so the live test now backs off.

Each of these has a regression test.
