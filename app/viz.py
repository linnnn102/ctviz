"""Turn an aggregation Result into a frontend-ready Visualization spec."""

from __future__ import annotations

from typing import Any

from app import dimensions as dims
from app.analysis.aggregate import Result
from app.plan import QueryPlan, SearchFilters
from app.schemas import Encoding, FieldEncoding, NetworkEncoding, Visualization, VisualizationOptions
from app.vocab import PHASE_LABELS, STATUS_LABELS

COUNT = FieldEncoding(field="trial_count", type="quantitative", title="Number of trials", unit="trials",
                      aggregate="sum")


def describe_filters(f: SearchFilters) -> str:
    parts: list[str] = []
    for name in ("intervention", "condition", "sponsor", "term"):
        if v := getattr(f, name):
            parts.append(v[:1].upper() + v[1:])
    if f.country:
        parts.append(f.country)
    if f.phases:
        parts.append(", ".join(PHASE_LABELS[p] for p in f.phases))
    if f.statuses:
        parts.append(", ".join(STATUS_LABELS[s] for s in f.statuses))
    if f.study_type:
        parts.append(f.study_type.title())
    if f.intervention_types:
        parts.append(", ".join(t.replace("_", " ").title() for t in f.intervention_types))
    if f.sponsor_classes:
        parts.append("Sponsor class: " + ", ".join(f.sponsor_classes))
    if f.start_year_min or f.start_year_max:
        if f.start_year_min and f.start_year_max:
            parts.append(f"started {f.start_year_min}–{f.start_year_max}"
                         if f.start_year_min != f.start_year_max else f"started {f.start_year_min}")
        elif f.start_year_min:
            parts.append(f"started {f.start_year_min} or later")
        else:
            parts.append(f"started by {f.start_year_max}")
    return " · ".join(parts) if parts else "All registered trials"


def _cat(dim: str, order: list[str] | None) -> FieldEncoding:
    d = dims.get(dim)
    return FieldEncoding(field=dim, type="ordinal" if d.order else "nominal", title=d.title, sort=order)


def _series(field: str | None, order: list[str] | None) -> FieldEncoding | None:
    if not field:
        return None
    title = "Cohort" if field == "cohort" else dims.get(field).title
    return FieldEncoding(field=field, type="nominal", title=title, sort=order)


def _orientation(order: list[str] | None) -> str:
    if not order:
        return "vertical"
    return "horizontal" if len(order) > 8 or max(len(o) for o in order) > 14 else "vertical"


def _table_columns(data: Any) -> list[FieldEncoding]:
    rows = data if isinstance(data, list) else data.get("nodes", [])
    if not rows:
        return []
    cols = []
    for k, v in rows[0].items():
        if k in {"citations", "citation_count", "nct_ids", "overflow"}:
            continue
        t = "quantitative" if isinstance(v, (int, float)) and not isinstance(v, bool) else "nominal"
        cols.append(FieldEncoding(field=k, type=t, title=k.replace("_", " ").capitalize()))
    return cols


def build(plan: QueryPlan, res: Result, cohort_labels: list[str], subject: str) -> Visualization:
    intent, chart = plan.intent, plan.chart_type
    multi = len(cohort_labels) > 1
    compare = " vs ".join(cohort_labels) if multi else None
    who = compare or subject

    if intent == "count":
        if multi:
            vis = Visualization(
                type="bar_chart", title=f"Number of trials: {compare}", subtitle=subject,
                encoding=Encoding(x=FieldEncoding(field="cohort", type="nominal", title="Cohort", sort=cohort_labels),
                                  y=COUNT),
                data=res.data, options=VisualizationOptions(orientation="vertical"))
        else:
            vis = Visualization(
                type="metric", title=f"Number of trials — {subject}",
                encoding=Encoding(value=COUNT, label=FieldEncoding(field="cohort", type="nominal")),
                data=res.data)

    elif intent == "distribution":
        g = plan.group_by
        d = dims.get(g)
        series_field = plan.series_by or ("cohort" if multi else None)
        if series_field:
            stacked = chart == "stacked_bar_chart"
            title = (f"Trials by {d.title.lower()}: {compare}" if multi
                     else f"Trials by {d.title.lower()} and {dims.get(series_field).title.lower()} — {subject}")
            vis = Visualization(
                type="stacked_bar_chart" if stacked else "grouped_bar_chart", title=title,
                subtitle=subject if multi else None,
                encoding=Encoding(x=_cat(g, res.category_order), y=COUNT, series=_series(series_field, res.series_order)),
                data=res.data,
                options=VisualizationOptions(orientation=_orientation(res.category_order), stacked=stacked))
        else:
            vis = Visualization(
                type="bar_chart", title=f"Trials by {d.title.lower()} — {subject}",
                encoding=Encoding(x=_cat(g, res.category_order), y=COUNT),
                data=res.data, options=VisualizationOptions(orientation=_orientation(res.category_order)))

    elif intent == "time_trend":
        g = plan.group_by
        sf = res.extra.get("series_field")
        verb = {"start_year": "started", "completion_year": "completing", "first_posted_year": "first posted"}[g]
        title = f"Trials {verb} per year — {who}"
        if plan.series_by:
            title = f"Trials {verb} per year by {dims.get(plan.series_by).title.lower()} — {subject}"
        bar = chart in {"bar_chart", "grouped_bar_chart", "stacked_bar_chart"}
        typ = "time_series"
        if bar:
            typ = "bar_chart" if not sf else ("stacked_bar_chart" if chart == "stacked_bar_chart" else "grouped_bar_chart")
        vis = Visualization(
            type=typ, title=title, subtitle=subject if multi else None,
            encoding=Encoding(x=FieldEncoding(field="year", type="temporal", title=dims.get(g).title, sort="ascending"),
                              y=COUNT, series=_series(sf, res.series_order)),
            data=res.data,
            options=VisualizationOptions(time_granularity="year", stacked=chart == "stacked_bar_chart"))

    elif intent == "histogram":
        d = dims.get(plan.group_by)
        vis = Visualization(
            type="histogram", title=f"Distribution of {d.title.lower()} — {who}",
            encoding=Encoding(
                x=FieldEncoding(field="bin_start", type="quantitative", title=d.title, unit=d.unit),
                x2=FieldEncoding(field="bin_end", type="quantitative", unit=d.unit),
                y=COUNT, label=FieldEncoding(field="bin_label", type="ordinal")),
            data=res.data)

    elif intent == "scatter":
        dx, dy = dims.get(plan.x), dims.get(plan.y)
        vis = Visualization(
            type="scatter_plot", title=f"{dy.title} vs {dx.title.lower()} — {who}",
            encoding=Encoding(
                x=FieldEncoding(field=plan.x, type="quantitative", title=dx.title, unit=dx.unit),
                y=FieldEncoding(field=plan.y, type="quantitative", title=dy.title, unit=dy.unit),
                series=_series(res.extra.get("series_field"), None),
                label=FieldEncoding(field="title", type="nominal")),
            data=res.data,
            options=VisualizationOptions(x_scale="log" if res.extra.get("log_x") else "linear",
                                         y_scale="log" if res.extra.get("log_y") else "linear"))

    elif intent == "network":
        net = plan.network
        a, b = dims.get(net.source).title, dims.get(net.target).title
        rel = f"{a} co-occurrence" if net.source == net.target else f"{a} ↔ {b}"
        vis = Visualization(
            type="network_graph", title=f"{rel} network — {who}",
            subtitle=("Edge weight = number of trials in which both appear" if net.source == net.target
                      else f"Edge weight = number of trials linking the {a.lower()} and {b.lower()}"),
            encoding=Encoding(network=NetworkEncoding()),
            data=res.data,
            options=VisualizationOptions(layout="force", bipartite=net.source != net.target))

    else:  # list
        vis = Visualization(type="table", title=f"Matching trials — {who}",
                            encoding=Encoding(columns=_table_columns(res.data)), data=res.data)

    if chart == "table" and vis.type != "table":
        rows = vis.data if isinstance(vis.data, list) else vis.data["edges"]
        vis = Visualization(type="table", title=vis.title, subtitle=vis.subtitle,
                            encoding=Encoding(columns=_table_columns(rows)), data=rows)
    return vis
