"""Plan validation and repair.

Whatever produced the plan (LLM or rules), this module makes it safe to run:

1. Grounding -- every free-text entity in the plan must be traceable to the
   user's question or structured fields; ungrounded ones are dropped.
2. Structured request fields override the plan (the caller is authoritative).
3. Intent/dimension compatibility is enforced, with documented defaults.
4. Every change is recorded in `adjustments` and surfaced in the response.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from app import dimensions
from app.plan import Cohort, NetworkSpec, QueryPlan, SearchFilters
from app.planner import lexicon as lx
from app.schemas import VisualizeRequest

ENTITY_FIELDS = ("condition", "intervention", "sponsor", "term", "country")

CHART_COMPAT: dict[str, set[str]] = {
    "count": {"metric", "bar_chart"},
    "distribution": {"bar_chart", "grouped_bar_chart", "stacked_bar_chart", "table"},
    "time_trend": {"time_series", "bar_chart", "grouped_bar_chart", "stacked_bar_chart", "table"},
    "histogram": {"histogram", "table"},
    "scatter": {"scatter_plot", "table"},
    "network": {"network_graph", "table"},
    "list": {"table"},
}

COMPARE_ON_FIELD = {"drug": "intervention", "condition": "condition", "sponsor": "sponsor", "country": "country"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def is_grounded(value: str, sources: list[str]) -> bool:
    """True if `value` (or something very close to it) occurs in one of `sources`."""
    v = _norm(value)
    if not v:
        return False
    for src in sources:
        s = _norm(src)
        if v in s:
            return True
        # fuzzy: tolerate typos / pluralization against same-length n-grams
        words, n = s.split(), len(v.split())
        for i in range(max(1, len(words) - n + 1)):
            gram = " ".join(words[i : i + n])
            if gram and SequenceMatcher(None, v, gram).ratio() >= 0.85:
                return True
    # token level: every specific token must appear (fuzzily) in a source; generic words may be added,
    # so "Parkinson's disease" is grounded by "Parkinson's trials" and "Alzheimer disease" by "Alzheimer's".
    specific = [t for t in v.split() if t not in GENERIC_TOKENS and len(t) > 1]
    if specific:
        src_tokens = {t for src in sources for t in _norm(src).split()}
        if all(any(t == s or (len(t) >= 4 and SequenceMatcher(None, t, s).ratio() >= 0.85) for s in src_tokens)
               for t in specific):
            return True
    # country aliases: "US" in text grounds "United States"
    for alias, canonical in lx.COUNTRY_ALIASES.items():
        if canonical.lower() == value.lower() and any(re.search(rf"\b{re.escape(alias)}\b", src.lower())
                                                      for src in sources):
            return True
    return False


# Words a planner may legitimately add when naming an entity the user mentioned.
GENERIC_TOKENS = {"disease", "diseases", "disorder", "disorders", "syndrome", "condition", "conditions", "s",
                  "the", "of", "and", "type"}


def _ground_filters(f: SearchFilters, sources: list[str], where: str, adj: list[str]) -> SearchFilters:
    for name in ENTITY_FIELDS:
        val = getattr(f, name)
        if val and not is_grounded(val, sources):
            adj.append(f"dropped ungrounded {where} filter {name}={val!r} (not found in query or fields)")
            setattr(f, name, None)
    return f


# Words that describe trial design or the question itself, not an entity to search for.
DESIGN_TERMS = {"combination", "combinations", "combination studies", "combination therapy", "trial", "trials",
                "study", "studies", "clinical trials", "randomized", "interventional", "drug", "drugs"}
DRUG_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}


def _drop_redundant_filters(plan: QueryPlan, adj: list[str]) -> None:
    """Remove filters that only restate the question's structure (they narrow results for no reason)."""
    for where, f in [("global", plan.filters)] + [(f"cohort '{c.label}'", c.filters) for c in plan.cohorts]:
        if f.term and _norm(f.term) in DESIGN_TERMS:
            adj.append(f"removed {where} term filter {f.term!r} (describes trial design, not an entity)")
            f.term = None
    uses_drug_dim = "drug" in {plan.group_by, plan.series_by} or (
        plan.network is not None and "drug" in {plan.network.source, plan.network.target})
    f = plan.filters
    if uses_drug_dim and f.intervention_types and set(f.intervention_types) <= DRUG_TYPES:
        adj.append(f"removed intervention_types={f.intervention_types} (already implied by the drug dimension)")
        f.intervention_types = None


def _apply_request_fields(f: SearchFilters, req: VisualizeRequest, adj: list[str]) -> None:
    mapping = {
        "drug_name": "intervention",
        "condition": "condition",
        "sponsor": "sponsor",
        "country": "country",
        "trial_phase": "phases",
        "status": "statuses",
        "study_type": "study_type",
        "intervention_type": "intervention_types",
        "sponsor_class": "sponsor_classes",
        "start_year": "start_year_min",
        "end_year": "start_year_max",
    }
    for req_field, plan_field in mapping.items():
        val = getattr(req, req_field)
        if val is None:
            continue
        current = getattr(f, plan_field)
        if current is not None and current != val:
            adj.append(f"structured field {req_field}={val!r} overrides planner value {current!r}")
        setattr(f, plan_field, val)
    # a free-text `term` that merely repeats a structured entity is redundant
    if f.term and any(_norm(f.term) == _norm(v) for v in (req.drug_name, req.condition, req.sponsor) if v):
        f.term = None


def _kind(dim: str | None) -> str | None:
    return dimensions.get(dim).kind if dim else None


# Adjustments that mean the *planner* produced something wrong or incomplete (as opposed to the
# caller overriding it, or harmless normalization). The cascade escalates to the large model on these.
SEVERE_PREFIXES = (
    "dropped ungrounded",
    "dropped cohort",
    "no grouping dimension",
    "histogram needs",
    "scatter x defaulted",
    "scatter y defaulted",
    "network spec missing",
    "network source",
    "network target",
    "series_by ",
)


def severe_adjustments(adjustments: list[str]) -> list[str]:
    return [a for a in adjustments if a.startswith(SEVERE_PREFIXES)]


def finalize_plan(plan: QueryPlan, req: VisualizeRequest) -> tuple[QueryPlan, list[str]]:
    adj: list[str] = []
    plan = plan.model_copy(deep=True)
    sources = [req.query] + [v for v in (req.drug_name, req.condition, req.sponsor, req.country) if v]
    sources += req.compare or []

    # 1. grounding
    plan.filters = _ground_filters(plan.filters, sources, "global", adj)
    kept: list[Cohort] = []
    for c in plan.cohorts:
        c.filters = _ground_filters(c.filters, sources, f"cohort '{c.label}'", adj)
        if c.filters.is_empty():
            adj.append(f"dropped cohort '{c.label}' (no grounded filters left)")
        else:
            kept.append(c)
    plan.cohorts = kept
    _drop_redundant_filters(plan, adj)  # before structured fields, so explicit caller filters are kept

    # 2. structured fields
    _apply_request_fields(plan.filters, req, adj)
    if req.compare:
        field = COMPARE_ON_FIELD[req.compare_on]
        plan.cohorts = [Cohort(label=v, filters=SearchFilters(**{field: v})) for v in req.compare]
        if getattr(plan.filters, field):
            adj.append(f"cleared global {field} filter because `compare` varies it per cohort")
            setattr(plan.filters, field, None)

    # cohort hygiene
    if len(plan.cohorts) == 1:
        only = plan.cohorts[0]
        plan.filters = plan.filters.merged(only.filters)
        plan.cohorts = []
        adj.append(f"single cohort '{only.label}' merged into global filters")
    if len(plan.cohorts) > 6:
        adj.append(f"trimmed comparison to the first 6 of {len(plan.cohorts)} cohorts")
        plan.cohorts = plan.cohorts[:6]
    seen: set[str] = set()
    for c in plan.cohorts:
        base = c.label.strip() or "Cohort"
        label, i = base, 2
        while label.lower() in seen:
            label, i = f"{base} ({i})", i + 1
        c.label = label
        seen.add(label.lower())

    # years
    f = plan.filters
    if f.start_year_min and f.start_year_max and f.start_year_min > f.start_year_max:
        f.start_year_min, f.start_year_max = f.start_year_max, f.start_year_min
        adj.append("swapped start/end year (were reversed)")
    for attr in ("start_year_min", "start_year_max"):
        y = getattr(f, attr)
        if y is not None and not 1900 <= y <= 2100:
            setattr(f, attr, None)
            adj.append(f"ignored out-of-range year {attr}={y}")

    # 3. user overrides of dimension
    if req.group_by:
        k = _kind(req.group_by)
        if plan.intent == "network":
            adj.append("group_by ignored for network intent")
        else:
            new_intent = {"temporal": "time_trend", "numeric": "histogram", "categorical": "distribution"}[k]
            if plan.intent in {"count", "list", "scatter"} or new_intent != plan.intent:
                if plan.intent != new_intent:
                    adj.append(f"intent {plan.intent} -> {new_intent} to honor group_by={req.group_by}")
                plan.intent = new_intent
            plan.group_by = req.group_by

    # 4. intent / dimension compatibility
    _repair_dimensions(plan, adj)

    # chart type
    chart = req.chart_type or plan.chart_type
    if chart and chart not in CHART_COMPAT[plan.intent]:
        adj.append(f"chart_type {chart} is incompatible with intent {plan.intent}; using default")
        chart = None
    plan.chart_type = chart

    if "top_n" in req.model_fields_set or not plan.top_n:
        plan.top_n = req.top_n
    plan.top_n = max(3, min(plan.top_n, 100))
    return plan, adj


def _repair_dimensions(plan: QueryPlan, adj: list[str]) -> None:
    intent = plan.intent
    gk = _kind(plan.group_by)

    if intent == "distribution":
        if gk == "temporal":
            plan.intent = "time_trend"
            adj.append(f"group_by {plan.group_by} is temporal: intent distribution -> time_trend")
        elif gk == "numeric":
            plan.intent = "histogram"
            adj.append(f"group_by {plan.group_by} is numeric: intent distribution -> histogram")
        elif gk is None:
            plan.group_by = "phase"
            adj.append("no grouping dimension given; defaulting to phase")
    elif intent == "time_trend":
        if gk == "categorical" and not plan.series_by:
            plan.series_by = plan.group_by
            adj.append(f"moved categorical {plan.group_by} to series_by for time trend")
        if gk != "temporal":
            plan.group_by = "start_year"
    elif intent == "histogram":
        if gk != "numeric":
            plan.group_by = plan.x if _kind(plan.x) == "numeric" else "enrollment"
            adj.append(f"histogram needs a numeric dimension; using {plan.group_by}")
    elif intent == "scatter":
        if _kind(plan.x) != "numeric":
            plan.x = "duration_months" if plan.y != "duration_months" else "enrollment"
            adj.append(f"scatter x defaulted to {plan.x}")
        if _kind(plan.y) != "numeric" or plan.y == plan.x:
            plan.y = "enrollment" if plan.x != "enrollment" else "site_count"
            adj.append(f"scatter y defaulted to {plan.y}")
    elif intent == "network":
        ents = set(dimensions.entity_dims())
        if plan.network is None:
            adj.append("network spec missing; defaulting to drug<->drug")
        net = plan.network or NetworkSpec(source="drug", target="drug")
        if net.source not in ents:
            adj.append(f"network source {net.source} is not an entity; using drug")
            net.source = "drug"
        if net.target not in ents:
            adj.append(f"network target {net.target} is not an entity; using {net.source}")
            net.target = net.source
        plan.network = net
        plan.group_by = None
    elif intent in {"count", "list"}:
        plan.group_by = None

    if plan.series_by and plan.intent not in {"distribution", "time_trend", "scatter"}:
        adj.append(f"series_by ignored for intent {plan.intent}")
        plan.series_by = None
    if plan.series_by:
        if _kind(plan.series_by) != "categorical" or plan.series_by == plan.group_by:
            adj.append(f"series_by {plan.series_by} dropped (must be a different categorical dimension)")
            plan.series_by = None
        elif plan.cohorts:
            adj.append("series_by dropped: comparison cohorts already occupy the series channel")
            plan.series_by = None
