# Planner model evaluation

- Date: 2026-10-21T14:10
- Queries: 50 ([evals/queries.json](../../queries.json)); candidate repeats: 2
- Reference: **gpt-5.4** (effort `medium`), validated plans. Candidates at effort `low` (ignored by non-reasoning gpt-4.1-mini).
- Exact = every plan component matches the reference. Cascade = escalate to a second gpt-5.4 pass on error / low confidence / severe validator repair.

## Summary

| Model | Exact (solo) | Component acc. | Escalation rate | **Exact (cascade)** | $/request solo | **$/request cascade** | p50 latency solo | p50 latency cascade | Errors |
|---|---|---|---|---|---|---|---|---|---|
| gpt-5.4 (2nd pass = ceiling) |  98.0% |  99.7% | — | — | $0.00789 | — | 4.2s | — | 0 |
| gpt-5-mini |  93.0% (92.0%–94.0%) |  99.0% |   7.0% | ** 94.0%** | $0.00106 | **$0.00170** | 5.2s | 5.3s | 0 |
| gpt-4.1-mini |  86.0% (84.0%–88.0%) |  98.0% |  13.0% | ** 88.0%** | $0.00078 | **$0.00205** | 1.8s | 1.8s | 0 |
| gpt-5.4-nano |  78.0% (74.0%–82.0%) |  96.1% |  17.0% | ** 86.0%** | $0.00058 | **$0.00212** | 2.3s | 2.5s | 0 |

## Selection

**Chosen planner model: `gpt-5-mini`** — best cascaded exact-match = 94.0%; eligible (within 3 pp): gpt-5-mini; cheapest eligible = gpt-5-mini at $0.00170/request.

For scale: always calling gpt-5.4 costs $0.00789/request; the chosen cascade costs $0.00170/request (78% less).

## Accuracy by plan component (solo, mean over runs)

| Model | intent | dims | series_by | entities | enums | years | cohorts |
|---|---|---|---|---|---|---|---|
| gpt-5.4 | 100.0% | 100.0% | 100.0% |  98.0% | 100.0% | 100.0% | 100.0% |
| gpt-5-mini | 100.0% |  98.0% | 100.0% |  98.0% |  99.0% |  98.0% | 100.0% |
| gpt-5.4-nano |  98.0% |  93.0% | 100.0% |  88.0% |  98.0% |  98.0% |  98.0% |
| gpt-4.1-mini |  98.0% |  98.0% | 100.0% |  95.0% |  98.0% |  97.0% | 100.0% |

## Disagreements with the reference (solo, first run)

### gpt-5.4

- `trend-05`: entities: got `{'term': 'car t'}` vs ref `{'intervention': 'car t'}`

### gpt-5-mini

- `trend-05`: dims: got `start_year` vs ref `first_posted_year`
- `trend-07`: years: got `(2021, 2026)` vs ref `(2021, None)`
- `geo-04`: entities: got `{'condition': 'malaria', 'term': 'vaccine'}` vs ref `{'condition': 'malaria', 'intervention': 'vaccine'}`
- `scat-02`: entities: got `{'term': 'oncology'}` vs ref `{'condition': 'oncology'}`
- `trend-07`: years: got `(2022, 2026)` vs ref `(2021, None)`
- `net-03`: enums: got `{'study_type': 'INTERVENTIONAL'}` vs ref `{}`
- `hist-02`: dims: got `enrollment` vs ref `duration_months`

### gpt-5.4-nano

- `trend-05`: dims: got `start_year` vs ref `first_posted_year`; entities: got `{'term': 'car t'}` vs ref `{'intervention': 'car t'}`
- `trend-07`: entities: got `{}` vs ref `{'intervention': 'semaglutide'}`
- `dist-07`: entities: got `{}` vs ref `{'intervention': 'keytruda'}`
- `cmp-04`: years: got `(2018, None)` vs ref `(None, None)`; cohorts: got `[]` vs ref `['{"e": {"intervention": "semaglutide"}, "n": {}, "y": [2018, null]}', '{"e": {"intervention": "tirzepatide"}, "n": {}, "y": [2018, null]}']`
- `cmp-07`: cohorts: got `[]` vs ref `['{"e": {"intervention": "adalimumab"}, "n": {}, "y": [null, null]}', '{"e": {"intervention": "etanercept"}, "n": {}, "y": [null, null]}', '{"e": {"intervention": "infliximab"}, "n": {}, "y": [null, null]}']`
- `geo-04`: entities: got `{'term': 'malaria vaccine'}` vs ref `{'condition': 'malaria', 'intervention': 'vaccine'}`
- `net-03`: enums: got `{'study_type': 'INTERVENTIONAL'}` vs ref `{}`
- `net-04`: dims: got `('condition', 'drug')` vs ref `('condition', 'condition')`
- `hist-02`: entities: got `{'term': 'alzheimer'}` vs ref `{'condition': 'alzheimer'}`
- `hist-03`: dims: got `enrollment` vs ref `site_count`
- `count-01`: entities: got `{'term': 'semaglutide'}` vs ref `{'intervention': 'semaglutide'}`
- `count-02`: entities: got `{}` vs ref `{'intervention': 'nivolumab'}`
- `list-02`: intent: got `time_trend` vs ref `list`; dims: got `start_year` vs ref `None`
- `trend-02`: entities: got `{'term': 'pembrolizumab'}` vs ref `{'intervention': 'pembrolizumab'}`
- `trend-05`: dims: got `start_year` vs ref `first_posted_year`; entities: got `{'term': 'car t'}` vs ref `{'intervention': 'car t'}`
- `trend-07`: years: got `(2022, None)` vs ref `(2021, None)`
- `dist-03`: enums: got `{'phases': ('PHASE3',), 'study_type': 'INTERVENTIONAL'}` vs ref `{'phases': ('PHASE3',)}`
- `dist-06`: entities: got `{}` vs ref `{'condition': 'glioblastoma'}`
- `dist-09`: intent: got `count` vs ref `distribution`; dims: got `None` vs ref `has_results`
- `geo-04`: entities: got `{'term': 'malaria vaccine'}` vs ref `{'condition': 'malaria', 'intervention': 'vaccine'}`
- `hist-02`: dims: got `enrollment` vs ref `duration_months`
- `count-01`: entities: got `{}` vs ref `{'intervention': 'semaglutide'}`

### gpt-4.1-mini

- `trend-05`: dims: got `start_year` vs ref `first_posted_year`
- `trend-06`: years: got `(2010, 2025)` vs ref `(None, None)`
- `trend-07`: years: got `(2021, 2026)` vs ref `(2021, None)`
- `geo-04`: entities: got `{'condition': 'malaria'}` vs ref `{'condition': 'malaria', 'intervention': 'vaccine'}`
- `net-03`: enums: got `{'study_type': 'INTERVENTIONAL'}` vs ref `{}`
- `typo-01`: entities: got `{'intervention': 'pembrolizumb'}` vs ref `{'intervention': 'pembrolizumab'}`
- `robust-02`: entities: got `{'condition': 'heart failure', 'intervention': 'device'}` vs ref `{'condition': 'heart failure'}`
- `robust-03`: intent: got `count` vs ref `list`
- `trend-05`: dims: got `start_year` vs ref `first_posted_year`
- `trend-07`: years: got `(2021, 2026)` vs ref `(2021, None)`
- `geo-04`: entities: got `{'condition': 'malaria'}` vs ref `{'condition': 'malaria', 'intervention': 'vaccine'}`
- `net-03`: enums: got `{'study_type': 'INTERVENTIONAL'}` vs ref `{}`
- `typo-01`: entities: got `{'intervention': 'pembrolizumb'}` vs ref `{'intervention': 'pembrolizumab'}`
- `robust-03`: intent: got `count` vs ref `list`

