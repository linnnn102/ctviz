"""Execution stage: validated QueryPlan -> API calls -> aggregation -> spec."""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone

from app import dimensions as dims
from app import viz
from app.analysis import aggregate as agg
from app.config import settings
from app.ctgov.client import CTGovClient, FetchResult
from app.ctgov.query import to_api_params
from app.dimensions import ExtractContext
from app.plan import QueryPlan, SearchFilters
from app.planner import PlanResult, make_plan
from app.schemas import CohortCoverage, Interpretation, ResponseMeta, VisualizeRequest, VisualizeResponse


def _dims_used(plan: QueryPlan) -> list[str]:
    used = [plan.group_by, plan.series_by, plan.x, plan.y]
    if plan.network:
        used += [plan.network.source, plan.network.target]
    if plan.intent == "list":
        used += ["phase", "status", "start_year", "sponsor", "enrollment"]
    return [d for d in used if d]


def _fields_for(plan: QueryPlan) -> list[str]:
    fields = list(dims.BASE_FIELDS)
    for d in _dims_used(plan):
        fields += [f for f in dims.get(d).api_fields if f not in fields]
    return fields


def _counting_rule(plan: QueryPlan) -> str:
    if plan.intent == "count":
        return "Exact number of trials matching the filters (API totalCount)."
    if plan.intent == "network":
        return ("Node trial_count = distinct trials listing the entity; edge weight = distinct trials in which both "
                "entities appear. Each trial is counted once even if it lists an entity several times.")
    if plan.intent in {"scatter", "list"}:
        return "One row per trial."
    if plan.intent == "histogram":
        return "Each trial with a value is counted once, in the bin containing its value (bins are [start, end))."
    d = dims.get(plan.group_by)
    if d.multi:
        return (f"Each trial is counted once per distinct {d.title.lower()} it lists, so one trial can contribute to "
                f"several categories and category totals can exceed the number of trials. `percent` is the share of "
                f"analyzed trials in the cohort.")
    return f"Each trial is counted exactly once, under its {d.title.lower()}."


def _relaxed(f: SearchFilters) -> tuple[SearchFilters, str] | None:
    """Fallback when an entity-specific search area returns nothing: retry as free text."""
    for name in ("intervention", "condition", "sponsor"):
        if val := getattr(f, name):
            g = f.model_copy()
            setattr(g, name, None)
            g.term = val if not g.term else f"{g.term} {val}"
            return g, f"no trials matched {name}={val!r}; retried as a free-text search"
    return None


class Engine:
    def __init__(self, client: CTGovClient):
        self.client = client

    async def _fetch(self, f: SearchFilters, fields: list[str], max_studies: int, sort: str | None,
                     warnings: list[str]) -> tuple[SearchFilters, FetchResult]:
        params = to_api_params(f)
        if sort:
            params["sort"] = sort
        res = await self.client.search(params, fields, max_studies)
        if res.total_count == 0 and (relaxed := _relaxed(f)):
            g, msg = relaxed
            params = to_api_params(g)
            if sort:
                params["sort"] = sort
            res2 = await self.client.search(params, fields, max_studies)
            if res2.total_count:
                warnings.append(msg)
                return g, res2
        return f, res

    async def run(self, req: VisualizeRequest, plan_result: PlanResult | None = None) -> VisualizeResponse:
        t0 = time.perf_counter()
        pr = plan_result or await make_plan(req)
        plan = pr.plan
        warnings = list(pr.warnings)
        notes: list[str] = []

        cohort_specs = (
            [(c.label, plan.filters.merged(c.filters)) for c in plan.cohorts]
            if plan.cohorts else [(viz.describe_filters(plan.filters), plan.filters)]
        )
        if plan.filters.is_empty() and not plan.cohorts:
            warnings.append("no filters were identified; results cover the first records of the entire registry")

        fields = _fields_for(plan)
        max_studies = min(req.max_studies or settings.default_max_studies, settings.hard_max_studies)
        sort = None
        if plan.intent == "count":
            max_studies = max(req.max_citations_per_datum, 1)
        elif plan.intent == "list":
            max_studies = min(max(plan.top_n, 25), 100)
            sort = "StartDate:desc"

        fetched = await asyncio.gather(*[
            self._fetch(f, fields, max_studies, sort, warnings) for _, f in cohort_specs
        ])

        aliases = {f.intervention.lower(): f.intervention.lower()
                   for f, _ in fetched if f.intervention}
        opts = agg.Opts(cap=req.max_citations_per_datum, include_ids=req.include_nct_ids, top_n=plan.top_n,
                        ctx=ExtractContext(intervention_aliases=aliases))
        cohorts = [agg.Cohort(label, r.studies, r.total_count) for (label, _), (_, r) in zip(cohort_specs, fetched)]

        coverage = []
        for (label, _), (f, r) in zip(cohort_specs, fetched):
            coverage.append(CohortCoverage(label=label, total_matching=r.total_count,
                                           analyzed=len(r.studies),
                                           truncated=r.truncated and plan.intent != "count",
                                           api_params=r.params))
            if r.total_count == 0:
                warnings.append(f"cohort '{label}' matched no trials")
            elif r.truncated and plan.intent not in {"count", "list"}:
                warnings.append(f"cohort '{label}': analyzed {len(r.studies)} of {r.total_count} matching trials "
                                f"(raise max_studies up to {settings.hard_max_studies} for full coverage)")

        res = self._dispatch(plan, cohorts, opts)
        notes += res.notes
        warnings += res.warnings

        subject = viz.describe_filters(plan.filters)
        vis = viz.build(plan, res, [c.label for c in cohorts] if plan.cohorts else [], subject)

        units = {"trial_count": "trials", "percent": "% of analyzed trials in cohort"}
        for d in _dims_used(plan):
            if u := dims.get(d).unit:
                units[d] = u
        if plan.intent == "histogram":
            units["bin_start"] = units["bin_end"] = dims.get(plan.group_by).unit or ""
        if "stats" in res.extra:
            notes.append("summary statistics: " + ", ".join(f"{k}={v:g}" for k, v in res.extra["stats"].items()))

        ts = await self.client.data_timestamp()
        meta = ResponseMeta(
            query=req.query,
            filters=plan.filters.model_dump(exclude_none=True),
            interpretation=Interpretation(planner=pr.planner, intent=plan.intent, rationale=plan.rationale or None,
                                          model=pr.model, escalated=pr.escalated,
                                          escalation_reason=pr.escalation_reason,
                                          plan=plan.model_dump(exclude_none=True), adjustments=pr.adjustments,
                                          llm_trace=pr.trace),
            coverage=coverage,
            counting_rule=_counting_rule(plan),
            sort=_sort_note(plan),
            units=units,
            warnings=warnings,
            notes=notes,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            data_as_of=date.fromisoformat(ts[:10]) if ts else None,
        )
        return VisualizeResponse(visualization=vis, meta=meta)

    @staticmethod
    def _dispatch(plan: QueryPlan, cohorts: list[agg.Cohort], opts: agg.Opts) -> agg.Result:
        f = plan.filters
        match plan.intent:
            case "count":
                return agg.counts(cohorts, opts)
            case "distribution":
                return agg.distribution(cohorts, plan.group_by, opts, plan.series_by)
            case "time_trend":
                return agg.time_trend(cohorts, plan.group_by, opts, plan.series_by,
                                      f.start_year_min if plan.group_by == "start_year" else None,
                                      f.start_year_max if plan.group_by == "start_year" else None)
            case "histogram":
                return agg.histogram(cohorts, plan.group_by, opts)
            case "scatter":
                return agg.scatter(cohorts, plan.x, plan.y, opts, plan.series_by)
            case "network":
                return agg.network(cohorts, plan.network.source, plan.network.target, opts)
            case "list":
                return agg.trial_table(cohorts, opts, limit=max(plan.top_n, 25))
        raise ValueError(f"unsupported intent {plan.intent}")


def _sort_note(plan: QueryPlan) -> str | None:
    if plan.intent == "distribution":
        d = dims.get(plan.group_by)
        return "fixed ordinal order (encoding.x.sort)" if d.order else "trial_count descending (encoding.x.sort)"
    if plan.intent == "time_trend":
        return "year ascending; years with no trials are included with trial_count=0"
    if plan.intent == "list":
        return "start date descending"
    if plan.intent == "network":
        return "nodes by trial_count descending; edges by weight descending"
    return None
