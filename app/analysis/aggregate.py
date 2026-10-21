"""Pure aggregation functions: study records -> rows/graphs with citations.

No I/O and no LLM here; everything is deterministic and unit-tested.
Each output datum carries:
  - trial_count       distinct trials contributing to the datum
  - citations         up to `cap` Citation dicts (nct_id, url, title, excerpts)
  - citation_count    == trial_count (how many trials *could* be cited)
  - nct_ids           (optional) every contributing NCT id
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations, product
from statistics import median
from typing import Any

from app import dimensions as dims
from app.dimensions import ExtractContext, Value

STUDY_URL = "https://clinicaltrials.gov/study/{}"


@dataclass
class Cohort:
    label: str
    studies: list[dict[str, Any]]
    total_matching: int


@dataclass
class Opts:
    cap: int = 5                  # citations per datum
    include_ids: bool = False
    top_n: int = 15
    ctx: ExtractContext = field(default_factory=ExtractContext)


def nct_of(study: dict) -> str:
    return dims.dig(study, f"{dims.ID}.nctId") or "UNKNOWN"


def title_of(study: dict) -> str | None:
    return dims.dig(study, f"{dims.ID}.briefTitle")


def citation(study: dict, excerpts: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    nid = nct_of(study)
    return {
        "nct_id": nid,
        "url": STUDY_URL.format(nid),
        "title": title_of(study),
        "excerpts": [{"field": f, "text": t} for f, t in excerpts]
        or [{"field": f"{dims.ID}.nctId", "text": nid}],
    }


class Bucket:
    __slots__ = ("key", "labels", "ids", "citations")

    def __init__(self, key: Any):
        self.key = key
        self.labels: Counter[str] = Counter()
        self.ids: dict[str, None] = {}  # ordered set
        self.citations: list[dict[str, Any]] = []

    def add(self, study: dict, label: str, excerpts: tuple[tuple[str, str], ...], cap: int) -> None:
        nid = nct_of(study)
        self.labels[label] += 1
        if nid in self.ids:
            return
        self.ids[nid] = None
        if len(self.citations) < cap:
            self.citations.append(citation(study, excerpts))

    @property
    def count(self) -> int:
        return len(self.ids)

    def payload(self, opts: Opts) -> dict[str, Any]:
        out: dict[str, Any] = {"trial_count": self.count, "citation_count": self.count,
                               "citations": self.citations}
        if opts.include_ids:
            out["nct_ids"] = list(self.ids)
        return out


def pick_label(dim: str, b: Bucket) -> str:
    """Display label for a bucket: most common raw spelling, or a prettified key
    when raw spellings are noisy (drug names with doses etc.)."""
    if dim in {"drug", "intervention"} and isinstance(b.key, str):
        exact = [lbl for lbl, _ in b.labels.most_common() if lbl.strip().lower() == b.key and not lbl.islower()]
        if exact:
            return exact[0].strip()
        return b.key if b.key.isupper() else " ".join(w[:1].upper() + w[1:] for w in b.key.split())
    return b.labels.most_common(1)[0][0] if b.labels else str(b.key)


def group(studies: list[dict], dim: str, opts: Opts) -> dict[Any, Bucket]:
    d = dims.get(dim)
    buckets: dict[Any, Bucket] = {}
    for s in studies:
        for v in d.extract(s, opts.ctx):
            buckets.setdefault(v.key, Bucket(v.key)).add(s, v.label, v.excerpts, opts.cap)
    return buckets


def _ordered_keys(dim: str, buckets: dict[Any, Bucket], top_n: int | None) -> tuple[list[Any], int]:
    """Sort keys (fixed ordinal order if defined, else count desc) and cut to top_n.
    Returns (keys, number_of_hidden_categories)."""
    d = dims.get(dim)
    if d.order:
        rank = {lbl: i for i, lbl in enumerate(d.order)}
        keys = sorted(buckets, key=lambda k: (rank.get(pick_label(dim, buckets[k]), 999), str(k)))
        return keys, 0
    keys = sorted(buckets, key=lambda k: (-buckets[k].count, pick_label(dim, buckets[k]).lower()))
    if top_n and len(keys) > top_n:
        return keys[:top_n], len(keys) - top_n
    return keys, 0


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total else 0.0


# ------------------------------------------------------------ analyses ---


@dataclass
class Result:
    data: Any
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    category_order: list[str] | None = None
    series_order: list[str] | None = None


def distribution(cohorts: list[Cohort], dim: str, opts: Opts, series_dim: str | None = None) -> Result:
    res = Result(data=[])
    if len(cohorts) == 1 and not series_dim:
        c = cohorts[0]
        buckets = group(c.studies, dim, opts)
        keys, hidden = _ordered_keys(dim, buckets, opts.top_n)
        n = len(c.studies)
        for k in keys:
            b = buckets[k]
            res.data.append({dim: pick_label(dim, b), "percent": _pct(b.count, n), **b.payload(opts)})
        if hidden:
            res.notes.append(f"showing top {len(keys)} of {len(keys) + hidden} distinct {dim} values")
        res.category_order = [r[dim] for r in res.data]
        return res

    # two-level: (category x series) where series = cohort or series_dim
    if series_dim:
        c = cohorts[0]
        sub_cohorts: dict[str, list[dict]] = {}
        sbuckets = group(c.studies, series_dim, opts)
        skeys, shidden = _ordered_keys(series_dim, sbuckets, 8)
        for sk in skeys:
            ids = sbuckets[sk].ids
            sub_cohorts[pick_label(series_dim, sbuckets[sk])] = [s for s in c.studies if nct_of(s) in ids]
        if shidden:
            res.notes.append(f"series limited to top {len(skeys)} {series_dim} values")
        series_list = [Cohort(lbl, studies, len(studies)) for lbl, studies in sub_cohorts.items()]
    else:
        series_list = cohorts

    per_series = [(sc, group(sc.studies, dim, opts)) for sc in series_list]
    combined: dict[Any, Bucket] = {}
    for _, bks in per_series:
        for k, b in bks.items():
            agg = combined.setdefault(k, Bucket(k))
            agg.labels.update(b.labels)
            for nid in b.ids:
                agg.ids[nid] = None
    keys, hidden = _ordered_keys(dim, combined, opts.top_n)
    if hidden:
        res.notes.append(f"showing top {len(keys)} of {len(keys) + hidden} distinct {dim} values (ranked by "
                         f"combined count)")
    series_field = series_dim or "cohort"
    for k in keys:
        label = pick_label(dim, combined[k])
        for sc, bks in per_series:
            b = bks.get(k) or Bucket(k)
            res.data.append({dim: label, series_field: sc.label, "percent": _pct(b.count, len(sc.studies)),
                             **b.payload(opts)})
    res.category_order = [pick_label(dim, combined[k]) for k in keys]
    res.series_order = [sc.label for sc in series_list]
    return res


def time_trend(cohorts: list[Cohort], dim: str, opts: Opts, series_dim: str | None,
               year_min: int | None, year_max: int | None) -> Result:
    res = Result(data=[])
    if series_dim:
        base = cohorts[0]
        sb = group(base.studies, series_dim, opts)
        skeys, hidden = _ordered_keys(series_dim, sb, 8)
        if hidden:
            res.notes.append(f"series limited to top {len(skeys)} {series_dim} values")
        series = [Cohort(pick_label(series_dim, sb[k]), [s for s in base.studies if nct_of(s) in sb[k].ids], 0)
                  for k in skeys]
        series_field = series_dim
    else:
        series = cohorts
        series_field = "cohort" if len(cohorts) > 1 else None

    grouped = [(sc, group(sc.studies, dim, opts)) for sc in series]
    years = [y for _, g in grouped for y in g]
    if not years:
        res.warnings.append(f"no trials have a usable {dim}")
        return res
    lo = max(min(years), year_min) if year_min else min(years)
    hi = min(max(years), year_max) if year_max else max(years)
    this_year = date.today().year
    if hi > this_year:
        res.notes.append(f"years after {this_year} reflect planned (anticipated) start dates")
    missing = sum(len(sc.studies) - sum(b.count for b in g.values()) for sc, g in grouped)
    if missing > 0 and dims.get(dim).multi is False:
        res.notes.append(f"{missing} trial(s) without a {dim.replace('_', ' ')} are excluded")
    for sc, g in grouped:
        for y in range(lo, hi + 1):
            b = g.get(y) or Bucket(y)
            row = {"year": y, **b.payload(opts)}
            if series_field:
                row[series_field] = sc.label
            res.data.append(row)
    res.series_order = [sc.label for sc in series] if series_field else None
    res.extra["series_field"] = series_field
    return res


def _nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * exp:
            return m * exp
    return 10 * exp


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def histogram(cohorts: list[Cohort], dim: str, opts: Opts) -> Result:
    res = Result(data=[])
    d = dims.get(dim)
    points: list[tuple[float, dict, Value]] = []
    for c in cohorts:
        for s in c.studies:
            for v in d.extract(s, opts.ctx):
                points.append((float(v.key), s, v))
    if not points:
        res.warnings.append(f"no trials have a numeric {dim}")
        return res
    vals = sorted(p[0] for p in points)
    n = len(vals)
    p95 = vals[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    lo = 0.0 if vals[0] >= 0 else vals[0]
    target_bins = max(5, min(20, int(math.sqrt(n))))
    step = _nice_step((p95 - lo) / target_bins or 1)
    edge = lo + step * math.ceil((p95 - lo) / step or 1)
    nbins = int(round((edge - lo) / step))
    overflow = vals[-1] > edge
    buckets = [Bucket(i) for i in range(nbins + (1 if overflow else 0))]
    for x, s, v in points:
        i = nbins if x > edge else min(int((x - lo) // step), nbins - 1)
        buckets[i].add(s, v.label, v.excerpts, opts.cap)
    for i, b in enumerate(buckets):
        start = lo + i * step
        if i == nbins:
            row = {"bin_start": edge, "bin_end": vals[-1], "bin_label": f"> {_fmt(edge)}", "overflow": True}
        else:
            end = start + step
            row = {"bin_start": start, "bin_end": end, "bin_label": f"{_fmt(start)}–{_fmt(end)}", "overflow": False}
        res.data.append({**row, **b.payload(opts)})
    res.extra["stats"] = {"n": n, "min": vals[0], "median": median(vals), "mean": round(sum(vals) / n, 1),
                          "p95": p95, "max": vals[-1], "bin_width": step}
    if overflow:
        res.notes.append(f"values above the 95th percentile ({_fmt(edge)}) are collected in an overflow bin")
    if n < sum(len(c.studies) for c in cohorts):
        res.notes.append(f"{sum(len(c.studies) for c in cohorts) - n} trial(s) lack {dim} and are excluded")
    return res


def scatter(cohorts: list[Cohort], x: str, y: str, opts: Opts, series_dim: str | None, max_points: int = 3000) -> Result:
    res = Result(data=[])
    dx, dy = dims.get(x), dims.get(y)
    ds = dims.get(series_dim) if series_dim else None
    series_field = series_dim or ("cohort" if len(cohorts) > 1 else None)
    skipped = 0
    for c in cohorts:
        for s in c.studies:
            vx, vy = dx.extract(s, opts.ctx), dy.extract(s, opts.ctx)
            if not vx or not vy:
                skipped += 1
                continue
            row = {"nct_id": nct_of(s), "title": title_of(s), x: vx[0].key, y: vy[0].key}
            excerpts = vx[0].excerpts + vy[0].excerpts
            if ds:
                sv = ds.extract(s, opts.ctx)
                row[series_dim] = sv[0].label if sv else "Unknown"
            elif series_field:
                row[series_field] = c.label
            row.update({"trial_count": 1, "citation_count": 1, "citations": [citation(s, excerpts)]})
            res.data.append(row)
    if len(res.data) > max_points:
        res.notes.append(f"plotting {max_points} of {len(res.data)} points")
        res.data = res.data[:max_points]
    if skipped:
        res.notes.append(f"{skipped} trial(s) missing {x} or {y} are not plotted")
    xs = [r[x] for r in res.data if r[x] > 0]
    ys = [r[y] for r in res.data if r[y] > 0]
    res.extra["log_x"] = bool(xs) and max(xs) / max(min(xs), 1) > 1000
    res.extra["log_y"] = bool(ys) and max(ys) / max(min(ys), 1) > 1000
    res.extra["series_field"] = series_field
    return res


def network(cohorts: list[Cohort], source: str, target: str, opts: Opts, max_edges: int | None = None) -> Result:
    res = Result(data={"nodes": [], "edges": []})
    ds, dt = dims.get(source), dims.get(target)
    same = source == target
    nodes: dict[str, Bucket] = {}
    node_type: dict[str, str] = {}
    edges: dict[tuple[str, str], Bucket] = {}
    seen: set[str] = set()

    def nid(t: str, v: Value) -> str:
        return f"{t}:{v.key}"

    for c in cohorts:
        for s in c.studies:
            if nct_of(s) in seen:  # cohorts may overlap; count each trial once
                continue
            seen.add(nct_of(s))
            a_vals = ds.extract(s, opts.ctx)
            b_vals = a_vals if same else dt.extract(s, opts.ctx)
            a_vals = [v for v in a_vals if v.key not in ("unknown", "Unknown")]
            b_vals = [v for v in b_vals if v.key not in ("unknown", "Unknown")]
            for t, vals in ((source, a_vals),) if same else ((source, a_vals), (target, b_vals)):
                for v in vals:
                    i = nid(t, v)
                    nodes.setdefault(i, Bucket(v.key)).add(s, v.label, v.excerpts, opts.cap)
                    node_type[i] = t
            pairs = combinations(sorted(a_vals, key=lambda v: str(v.key)), 2) if same else product(a_vals, b_vals)
            for va, vb in pairs:
                ia, ib = nid(source, va), nid(target, vb)
                if ia == ib:
                    continue
                k = (ia, ib) if same is False or ia < ib else (ib, ia)
                edges.setdefault(k, Bucket(k)).add(s, "", va.excerpts + vb.excerpts, opts.cap)

    # prune: top_n nodes per type among nodes that participate in an edge
    in_edge: set[str] = {n for e in edges for n in e}
    keep: set[str] = set()
    for t in {source, target}:
        typed = sorted((i for i in in_edge if node_type[i] == t), key=lambda i: (-nodes[i].count, i))
        keep.update(typed[: opts.top_n])
        if len(typed) > opts.top_n:
            res.notes.append(f"showing top {opts.top_n} of {len(typed)} connected {t} nodes (by trial count)")
    max_edges = max_edges or opts.top_n * 6
    kept_edges = sorted((e for e in edges if e[0] in keep and e[1] in keep), key=lambda e: (-edges[e].count, e))
    if len(kept_edges) > max_edges:
        res.notes.append(f"showing the {max_edges} strongest of {len(kept_edges)} edges")
        kept_edges = kept_edges[:max_edges]
    degree: Counter[str] = Counter()
    for a, b in kept_edges:
        degree[a] += 1
        degree[b] += 1
    alias_keys = set(opts.ctx.intervention_aliases.values())
    for i in sorted(keep, key=lambda i: (-nodes[i].count, i)):
        if not degree[i]:
            continue
        t = node_type[i]
        res.data["nodes"].append({
            "id": i, "label": pick_label(t, nodes[i]), "type": t, "degree": degree[i],
            "is_query_entity": t in {"drug", "intervention"} and nodes[i].key in alias_keys,
            **nodes[i].payload(opts),
        })
    for a, b in kept_edges:
        e = edges[(a, b)]
        res.data["edges"].append({"source": a, "target": b, "weight": e.count, **e.payload(opts)})
    if not res.data["edges"]:
        res.warnings.append("no relationships found among the analyzed trials" +
                            (" (trials with a single drug have no co-occurrence edges)" if same else ""))
    res.extra["trials_in_graph"] = len(seen)
    return res


def trial_table(cohorts: list[Cohort], opts: Opts, limit: int) -> Result:
    res = Result(data=[])
    cols = ["phase", "status", "start_year", "sponsor", "enrollment"]
    rows = []
    for c in cohorts:
        for s in c.studies:
            row: dict[str, Any] = {"nct_id": nct_of(s), "title": title_of(s)}
            excerpts: list[tuple[str, str]] = []
            for col in cols:
                vals = dims.get(col).extract(s, opts.ctx)
                row[col] = vals[0].label if vals else None
                if vals:
                    excerpts.extend(vals[0].excerpts)
            row["start_date"] = dims.dig(s, f"{dims.STATUS}.startDateStruct.date")
            if len(cohorts) > 1:
                row["cohort"] = c.label
            row.update({"trial_count": 1, "citation_count": 1, "citations": [citation(s, tuple(excerpts))]})
            rows.append(row)
    rows.sort(key=lambda r: r.get("start_date") or "", reverse=True)
    if len(rows) > limit:
        res.notes.append(f"showing {limit} most recently started of {len(rows)} analyzed trials")
    res.data = rows[:limit]
    return res


def counts(cohorts: list[Cohort], opts: Opts) -> Result:
    """Exact totals from the API's totalCount; citations sampled from fetched records."""
    res = Result(data=[])
    for c in cohorts:
        b = Bucket(c.label)
        for s in c.studies[: max(opts.cap, 1)]:
            b.add(s, c.label, ((f"{dims.ID}.nctId", nct_of(s)),), opts.cap)
        row = {"cohort": c.label, **b.payload(opts)}
        row["trial_count"] = c.total_matching
        row["citation_count"] = c.total_matching
        res.data.append(row)
    res.notes.append("trial_count is the API's exact totalCount; citations are a sample of matching trials")
    return res
