"""Engine + HTTP tests with ClinicalTrials.gov mocked (respx) and the LLM stubbed."""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import app.planner as planner_mod
from app.analysis.engine import Engine
from app.ctgov.client import CTGovClient
from app.plan import Cohort, NetworkSpec, QueryPlan, SearchFilters
from app.config import settings
from app.planner.llm import LLMCall, LLMPlan, PlannerError
from app.schemas import VisualizeRequest

API = "https://clinicaltrials.gov/api/v2"


def page(studies, total=None, token=None):
    body = {"studies": studies, "totalCount": total if total is not None else len(studies)}
    if token:
        body["nextPageToken"] = token
    return httpx.Response(200, json=body)


@pytest.fixture
def api():
    with respx.mock(base_url=API, assert_all_called=False) as mock:
        mock.get("/version").mock(return_value=httpx.Response(200, json={"dataTimestamp": "2026-10-19T09:00:00"}))
        yield mock


async def run(req: VisualizeRequest):
    client = CTGovClient()
    try:
        return await Engine(client).run(req)
    finally:
        await client.aclose()


async def test_distribution_end_to_end(api, studies):
    route = api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query="How are melanoma trials distributed across phases?", use_llm=False))
    v = resp.visualization
    assert v.type == "bar_chart"
    assert v.encoding.x.field == "phase" and v.encoding.y.field == "trial_count"
    assert v.encoding.x.sort == [r["phase"] for r in v.data]
    params = route.calls[0].request.url.params
    assert params["query.cond"] == "melanoma"
    assert "Phase" in params["fields"] and "NCTId" in params["fields"]
    assert resp.meta.coverage[0].total_matching == 4 and not resp.meta.coverage[0].truncated
    assert resp.meta.interpretation.planner == "rules"
    assert str(resp.meta.data_as_of) == "2026-10-19"


async def test_pagination_and_truncation_warning(api, studies):
    api.get("/studies", params={"pageToken": "p2"}).mock(return_value=page(studies[2:], total=10))
    api.get("/studies").mock(return_value=page(studies[:2], total=10, token="p2"))
    resp = await run(VisualizeRequest(query="melanoma trials by phase", max_studies=50, use_llm=False))
    cov = resp.meta.coverage[0]
    assert cov.analyzed == 4 and cov.total_matching == 10 and cov.truncated
    assert any("analyzed 4 of 10" in w for w in resp.meta.warnings)


async def test_retry_on_server_error(api, studies):
    api.get("/studies").mock(side_effect=[httpx.Response(503), page(studies)])
    resp = await run(VisualizeRequest(query="melanoma trials by phase", use_llm=False))
    assert resp.meta.coverage[0].analyzed == 4


async def test_zero_results_relaxes_to_free_text(api, studies):
    def handler(request):
        if "query.intr" in request.url.params:
            return page([], total=0)
        return page(studies)
    api.get("/studies").mock(side_effect=handler)
    resp = await run(VisualizeRequest(query="trials by phase", drug_name="obscurumab", use_llm=False))
    assert any("retried as a free-text search" in w for w in resp.meta.warnings)
    assert resp.meta.coverage[0].analyzed == 4


async def test_comparison_issues_one_query_per_cohort(api, studies):
    route = api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query="Compare phases for trials involving Pembrolizumab vs Nivolumab",
                                      use_llm=False))
    assert resp.visualization.type == "grouped_bar_chart"
    assert resp.visualization.encoding.series.field == "cohort"
    intr = sorted(c.request.url.params["query.intr"] for c in route.calls)
    assert intr == ["Nivolumab", "Pembrolizumab"]


async def test_network_spec(api, studies):
    api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query="drug ↔ drug network for melanoma", use_llm=False))
    v = resp.visualization
    assert v.type == "network_graph" and set(v.data) == {"nodes", "edges"}
    ids = {n["id"] for n in v.data["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in v.data["edges"])


async def test_count_metric_uses_total(api, studies):
    route = api.get("/studies").mock(return_value=page(studies[:1], total=169))
    resp = await run(VisualizeRequest(query="How many recruiting trials are there for semaglutide?", use_llm=False))
    assert resp.visualization.type == "metric"
    assert resp.visualization.data[0]["trial_count"] == 169
    assert route.calls[0].request.url.params["pageSize"] == "5"  # no need to download everything


SMALL, BIG = settings.planner_model, settings.escalation_model
GOOD = QueryPlan(intent="network", network=NetworkSpec(source="sponsor", target="drug"),
                 filters=SearchFilters(condition="melanoma"), rationale="sponsor-drug network")
HALLUCINATED = QueryPlan(intent="network", network=NetworkSpec(source="sponsor", target="drug"),
                         filters=SearchFilters(condition="melanoma", sponsor="Pfizer"))  # Pfizer not in question
NETWORK_Q = "Show a network of sponsors and drugs for melanoma trials"


def stub_models(monkeypatch, outcomes: dict):
    """outcomes: model -> QueryPlan | (QueryPlan, confidence) | Exception. Records call order."""
    calls: list[str] = []

    async def fake(req, model, effort=None):
        calls.append(model)
        out = outcomes[model]
        if isinstance(out, Exception):
            raise out
        plan, conf = out if isinstance(out, tuple) else (out, "high")
        return LLMCall(output=LLMPlan(plan=plan, confidence=conf, ambiguities=["unsure"] if conf == "low" else []),
                       model=model, latency_ms=10, input_tokens=2600, cached_tokens=0, output_tokens=400,
                       reasoning_tokens=200)

    monkeypatch.setattr(planner_mod, "llm_available", lambda: True)
    monkeypatch.setattr(planner_mod, "plan_with_llm", fake)
    return calls


async def test_cascade_accepts_clean_small_model_plan(api, studies, monkeypatch):
    calls = stub_models(monkeypatch, {SMALL: GOOD, BIG: AssertionError("must not be called")})
    api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query=NETWORK_Q))
    it = resp.meta.interpretation
    assert calls == [SMALL]
    assert (it.planner, it.model, it.escalated) == ("llm", SMALL, False)
    assert it.llm_trace[0]["outcome"] == "accepted" and it.llm_trace[0]["cost_usd"] > 0


async def test_cascade_escalates_on_ungrounded_entity(api, studies, monkeypatch):
    calls = stub_models(monkeypatch, {SMALL: HALLUCINATED, BIG: GOOD})
    route = api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query=NETWORK_Q))
    it = resp.meta.interpretation
    assert calls == [SMALL, BIG]
    assert it.model == BIG and it.escalated and "Pfizer" in it.escalation_reason
    assert [t["outcome"] for t in it.llm_trace] == ["escalated", "accepted"]
    assert "query.spons" not in route.calls[0].request.url.params


async def test_cascade_escalates_on_low_confidence(api, studies, monkeypatch):
    calls = stub_models(monkeypatch, {SMALL: (GOOD, "low"), BIG: GOOD})
    api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query=NETWORK_Q))
    assert calls == [SMALL, BIG]
    assert resp.meta.interpretation.escalation_reason.startswith("low confidence")


async def test_cascade_keeps_validated_small_plan_if_big_model_fails(api, studies, monkeypatch):
    stub_models(monkeypatch, {SMALL: HALLUCINATED, BIG: PlannerError("gpt-5.4 timed out")})
    route = api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query=NETWORK_Q))
    it = resp.meta.interpretation
    assert it.model == SMALL and not it.escalated
    assert "query.spons" not in route.calls[0].request.url.params  # still validated
    assert any("escalation model" in w for w in resp.meta.warnings)


async def test_cascade_falls_back_to_rules_when_all_models_fail(api, studies, monkeypatch):
    stub_models(monkeypatch, {SMALL: PlannerError("rate limited"), BIG: PlannerError("rate limited")})
    api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query="How are melanoma trials distributed across phases?"))
    assert resp.meta.interpretation.planner == "rules"
    assert any("rule-based planner" in w for w in resp.meta.warnings)
    assert [t["outcome"] for t in resp.meta.interpretation.llm_trace] == ["escalated", "failed"]


async def test_use_llm_false_skips_models(api, studies, monkeypatch):
    calls = stub_models(monkeypatch, {SMALL: GOOD, BIG: GOOD})
    api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query=NETWORK_Q, use_llm=False))
    assert calls == [] and resp.meta.interpretation.planner == "rules"


async def test_llm_cohorts_with_shared_filters(api, studies, monkeypatch):
    plan = QueryPlan(intent="time_trend", group_by="start_year", filters=SearchFilters(start_year_min=2015),
                     cohorts=[Cohort(label="Asthma", filters=SearchFilters(condition="asthma")),
                              Cohort(label="COPD", filters=SearchFilters(condition="COPD"))])
    stub_models(monkeypatch, {SMALL: plan, BIG: plan})
    route = api.get("/studies").mock(return_value=page(studies))
    resp = await run(VisualizeRequest(query="asthma vs COPD trials per year since 2015"))
    assert resp.visualization.type == "time_series"
    assert resp.visualization.encoding.series.field == "cohort"
    for call in route.calls:
        assert "RANGE[2015-01-01,MAX]" in call.request.url.params["filter.advanced"]


def test_http_layer(api, studies):
    api.get("/studies").mock(return_value=page(studies))
    from app.main import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/visualize", json={"query": "melanoma trials by phase", "use_llm": False})
        assert r.status_code == 200 and r.json()["visualization"]["type"] == "bar_chart"
        assert c.post("/visualize", json={"query": "x"}).status_code == 422
        assert c.post("/visualize", json={"query": "ok query", "trial_phase": ["PHASE9"]}).status_code == 422
        assert len(c.get("/dimensions").json()) > 15


def test_http_upstream_error_is_502(api):
    api.get("/studies").mock(return_value=httpx.Response(400, text="bad filter"))
    from app.main import app
    with TestClient(app) as c:
        r = c.post("/visualize", json={"query": "melanoma trials by phase", "use_llm": False})
        assert r.status_code == 502 and r.json()["error"] == "upstream_error"


# ------------------------------------------------------------ guardrails ---

def test_rate_limit_per_ip(api, studies, monkeypatch):
    import dataclasses

    import app.main as main_mod
    monkeypatch.setattr(main_mod, "settings", dataclasses.replace(settings, rate_limit_per_min=2))
    monkeypatch.setattr(main_mod, "rate_limiter", main_mod.RateLimiter())
    api.get("/studies").mock(return_value=page(studies))
    body = {"query": "melanoma trials by phase", "use_llm": False}
    with TestClient(main_mod.app) as c:
        a = {"x-forwarded-for": "1.1.1.1, 10.0.0.1"}
        assert c.post("/visualize", json=body, headers=a).status_code == 200
        assert c.post("/visualize", json=body, headers=a).status_code == 200
        r = c.post("/visualize", json=body, headers=a)
        assert r.status_code == 429 and r.json()["error"] == "rate_limited" and r.headers["retry-after"] == "60"
        assert c.post("/visualize", json=body, headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
        assert c.get("/health").status_code == 200  # GETs are never limited


async def test_daily_llm_budget_falls_back_to_rules(api, studies, monkeypatch):
    import dataclasses
    calls = stub_models(monkeypatch, {SMALL: GOOD, BIG: GOOD})
    monkeypatch.setattr(planner_mod, "settings", dataclasses.replace(settings, daily_llm_limit=1))
    monkeypatch.setattr(planner_mod, "llm_budget", planner_mod.DailyBudget())
    api.get("/studies").mock(return_value=page(studies))
    first = await run(VisualizeRequest(query=NETWORK_Q))
    second = await run(VisualizeRequest(query=NETWORK_Q))
    assert first.meta.interpretation.planner == "llm"
    assert second.meta.interpretation.planner == "rules"
    assert any("daily LLM budget" in w for w in second.meta.warnings)
    assert calls == [SMALL]
