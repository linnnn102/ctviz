"""Planner model evaluation: accuracy vs cost for the small-model tier of the cascade.

    uv run --env-file .env python evals/run_eval.py
    uv run --env-file .env python evals/run_eval.py --candidates gpt-5-mini gpt-4.1-mini --repeats 1

Method
------
* Reference ("truth"): gpt-5.4 (escalation effort) plans for every query in
  evals/queries.json, after the same validator the service uses.
* Ceiling: a second, independent gpt-5.4 pass. Its agreement with the
  reference measures gpt-5.4's own run-to-run noise -- no candidate can be
  expected to beat it. The cascade also uses this pass as the answer for
  escalated queries, so escalations are not trivially "correct".
* Candidates run at the planner effort, `--repeats` times each.
* Scoring compares the *executed* plans component by component:
  intent, primary dimensions, series_by, entity filters, enum filters, years, cohorts.
  exact = all components agree; component accuracy = mean over components.
* Cascade simulation: a query escalates when the candidate errors, says
  confidence=low, or needs severe validator repairs (same rule as production).

Selection rule (fixed before looking at results)
------------------------------------------------
Let B = best cascaded exact-match rate among candidates. Eligible = candidates
with cascaded exact >= B - 3 percentage points. Pick the eligible candidate
with the lowest expected cost per request (small call + escalations).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.plan import QueryPlan  # noqa: E402
from app.planner import escalation_reason, escalation_reason_from  # noqa: E402
from app.planner.llm import PlannerError, plan_with_llm  # noqa: E402
from app.planner.validate import finalize_plan  # noqa: E402
from app.schemas import VisualizeRequest  # noqa: E402

COMPONENTS = ["intent", "dims", "series_by", "entities", "enums", "years", "cohorts"]
ENTITY_FIELDS = ("condition", "intervention", "sponsor", "term", "country")
ENUM_FIELDS = ("phases", "statuses", "study_type", "intervention_types", "sponsor_classes")


# ------------------------------------------------------------------ scoring ---

def _norm(s: str) -> str:
    s = re.sub(r"'s\b", "", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _entities(f: dict) -> dict[str, str]:
    return {k: _norm(f[k]) for k in ENTITY_FIELDS if f.get(k)}


def _enums(f: dict) -> dict[str, Any]:
    out = {}
    for k in ENUM_FIELDS:
        v = f.get(k)
        if v:
            out[k] = tuple(sorted(v)) if isinstance(v, list) else v
    return out


def canonical(plan: dict) -> dict[str, Any]:
    f = plan.get("filters", {})
    intent = plan["intent"]
    if intent == "scatter":
        dims = (plan.get("x"), plan.get("y"))
    elif intent == "network":
        n = plan.get("network") or {}
        dims = tuple(sorted((n.get("source"), n.get("target"))))  # sponsor<->drug == drug<->sponsor
    elif intent in {"count", "list"}:
        dims = None
    else:
        dims = plan.get("group_by")
    # Compare what each cohort actually queries: shared filters merged into cohort filters, so
    # "breast cancer" placed globally or repeated inside each cohort scores the same.
    def eff(c: dict) -> dict:
        return {**{k: v for k, v in f.items() if v is not None},
                **{k: v for k, v in c["filters"].items() if v is not None}}

    cohorts = sorted(
        json.dumps({"e": _entities(e), "n": _enums(e), "y": (e.get("start_year_min"), e.get("start_year_max"))},
                   sort_keys=True)
        for e in (eff(c) for c in plan.get("cohorts", []))
    )
    folded = bool(cohorts)  # with cohorts, filters are fully described by `cohorts`
    return {
        "intent": intent,
        "dims": dims,
        "series_by": plan.get("series_by"),
        "entities": {} if folded else _entities(f),
        "enums": {} if folded else _enums(f),
        "years": (None, None) if folded else (f.get("start_year_min"), f.get("start_year_max")),
        "cohorts": cohorts,
    }


def _entities_match(a: dict[str, str], b: dict[str, str]) -> bool:
    """Same fields; values equal or one contains the other ('type 2 diabetes' ~ 'diabetes type 2' is NOT ok,
    'alzheimer disease' ~ 'alzheimer' is)."""
    if set(a) != set(b):
        return False
    return all(a[k] == b[k] or a[k] in b[k] or b[k] in a[k] for k in a)


def compare(pred: dict, ref: dict) -> dict[str, bool]:
    cp, cr = canonical(pred), canonical(ref)
    out = {}
    for comp in COMPONENTS:
        if comp == "entities":
            out[comp] = _entities_match(cp[comp], cr[comp])
        else:
            out[comp] = cp[comp] == cr[comp]
    return out


# ------------------------------------------------------------------ running ---

async def run_one(sem: asyncio.Semaphore, item: dict, model: str, effort: str | None) -> dict:
    req = VisualizeRequest(**item["request"])
    async with sem:
        try:
            call = await plan_with_llm(req, model, effort)
        except PlannerError as exc:
            return {"id": item["id"], "model": model, "error": str(exc)}
    plan, adj = finalize_plan(call.output.plan, req)
    return {
        "id": item["id"], "model": model,
        "plan": plan.model_dump(exclude_none=True), "raw_plan": call.output.plan.model_dump(exclude_none=True),
        "adjustments": adj, "confidence": call.output.confidence, "ambiguities": call.output.ambiguities,
        "escalate": escalation_reason(call, adj),
        "latency_ms": call.latency_ms, "input_tokens": call.input_tokens, "cached_tokens": call.cached_tokens,
        "output_tokens": call.output_tokens, "reasoning_tokens": call.reasoning_tokens, "cost_usd": call.cost_usd,
    }


async def run_pass(items: list[dict], model: str, effort: str | None, concurrency: int, tag: str) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[run_one(sem, it, model, effort) for it in items])
    errs = sum(1 for r in rows if "error" in r)
    print(f"  {tag:<22} {len(rows)} queries in {time.perf_counter() - t0:5.1f}s  errors={errs}", flush=True)
    return list(rows)


# --------------------------------------------------------------- reporting ---

def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def summarize(name: str, passes: list[list[dict]], ref: dict[str, dict], esc: dict[str, dict] | None) -> dict:
    exact, comp_acc, casc_exact, cost, casc_cost, lat, casc_lat, esc_rate, errors = ([] for _ in range(9))
    per_component = {c: [] for c in COMPONENTS}
    misses: list[dict] = []
    for rows in passes:
        ex = ca = cx = co = cc = er = n = errs = 0
        for r in rows:
            gold = ref.get(r["id"])
            if gold is None or "plan" not in gold:
                continue
            n += 1
            if "error" in r:
                errs += 1
                ok_solo, comps = False, {c: False for c in COMPONENTS}
            else:
                comps = compare(r["plan"], gold["plan"])
                ok_solo = all(comps.values())
                co += r["cost_usd"] or 0
                lat.append(r["latency_ms"])
            for c in COMPONENTS:
                per_component[c].append(comps[c])
            ex += ok_solo
            ca += sum(comps.values()) / len(COMPONENTS)
            if not ok_solo and len(misses) < 60:
                misses.append({"id": r["id"], "error": r.get("error"),
                               "pred": canonical(r["plan"]) if "plan" in r else None,
                               "ref": canonical(gold["plan"]),
                               "wrong": [c for c, v in comps.items() if not v]})
            # cascade
            if esc is not None:
                escalates = "error" in r or bool(r.get("escalate"))
                q_cost = r.get("cost_usd") or 0
                q_lat = r.get("latency_ms") or 0
                if escalates and r["id"] in esc and "plan" in esc[r["id"]]:
                    er += 1
                    final_ok = all(compare(esc[r["id"]]["plan"], gold["plan"]).values())
                    q_cost += esc[r["id"]]["cost_usd"] or 0
                    q_lat += esc[r["id"]]["latency_ms"]
                else:
                    final_ok = ok_solo
                cx += final_ok
                cc += q_cost
                casc_lat.append(q_lat)
        exact.append(ex / n)
        comp_acc.append(ca / n)
        cost.append(co / n)
        errors.append(errs)
        if esc is not None:
            casc_exact.append(cx / n)
            casc_cost.append(cc / n)
            esc_rate.append(er / n)
    mean = statistics.mean
    out = {
        "model": name, "runs": len(passes), "n": n,
        "exact": mean(exact), "exact_range": (min(exact), max(exact)),
        "component_acc": mean(comp_acc),
        "per_component": {c: mean(v) for c, v in per_component.items()},
        "cost_per_query": mean(cost),
        "latency_p50_ms": statistics.median(lat) if lat else None,
        "latency_p95_ms": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
        "errors": sum(errors),
        "misses": misses,
    }
    if esc is not None:
        out |= {"cascade_exact": mean(casc_exact), "cascade_cost_per_query": mean(casc_cost),
                "escalation_rate": mean(esc_rate),
                "cascade_latency_p50_ms": statistics.median(casc_lat) if casc_lat else None}
    return out


def choose(summaries: list[dict]) -> tuple[dict, str]:
    best = max(s["cascade_exact"] for s in summaries)
    eligible = [s for s in summaries if s["cascade_exact"] >= best - 0.03]
    pick = min(eligible, key=lambda s: s["cascade_cost_per_query"])
    why = (f"best cascaded exact-match = {pct(best).strip()}; eligible (within 3 pp): "
           f"{', '.join(s['model'] for s in eligible)}; cheapest eligible = {pick['model']} "
           f"at ${pick['cascade_cost_per_query']:.5f}/request")
    return pick, why


def write_report(out_dir: Path, ref_model: str, ceiling: dict, summaries: list[dict], pick: dict, why: str,
                 n_queries: int, args: argparse.Namespace) -> Path:
    big_cost = ceiling["cost_per_query"]
    lines = [
        "# Planner model evaluation", "",
        f"- Date: {datetime.now().isoformat(timespec='minutes')}",
        f"- Queries: {n_queries} ([evals/queries.json](../../queries.json)); candidate repeats: {args.repeats}",
        f"- Reference: **{ref_model}** (effort `{args.ref_effort}`), validated plans. Candidates at effort "
        f"`{args.effort}` (ignored by non-reasoning gpt-4.1-mini).",
        "- Exact = every plan component matches the reference. Cascade = escalate to a second gpt-5.4 pass on "
        "error / low confidence / severe validator repair.", "",
        "## Summary", "",
        "| Model | Exact (solo) | Component acc. | Escalation rate | **Exact (cascade)** | $/request solo | "
        "**$/request cascade** | p50 latency solo | p50 latency cascade | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| {ref_model} (2nd pass = ceiling) | {pct(ceiling['exact'])} | {pct(ceiling['component_acc'])} | — | — | "
        f"${big_cost:.5f} | — | {ceiling['latency_p50_ms'] / 1000:.1f}s | — | {ceiling['errors']} |",
    ]
    for s in sorted(summaries, key=lambda s: -s["cascade_exact"]):
        rng = f" ({pct(s['exact_range'][0]).strip()}–{pct(s['exact_range'][1]).strip()})" if s["runs"] > 1 else ""
        lines.append(
            f"| {s['model']} | {pct(s['exact'])}{rng} | {pct(s['component_acc'])} | {pct(s['escalation_rate'])} | "
            f"**{pct(s['cascade_exact'])}** | ${s['cost_per_query']:.5f} | **${s['cascade_cost_per_query']:.5f}** | "
            f"{s['latency_p50_ms'] / 1000:.1f}s | {s['cascade_latency_p50_ms'] / 1000:.1f}s | {s['errors']} |")
    lines += ["", "## Selection", "", f"**Chosen planner model: `{pick['model']}`** — {why}.", "",
              f"For scale: always calling {ref_model} costs ${big_cost:.5f}/request; the chosen cascade costs "
              f"${pick['cascade_cost_per_query']:.5f}/request "
              f"({100 * (1 - pick['cascade_cost_per_query'] / big_cost):.0f}% less).", "",
              "## Accuracy by plan component (solo, mean over runs)", "",
              "| Model | " + " | ".join(COMPONENTS) + " |", "|---|" + "---|" * len(COMPONENTS)]
    for s in [ceiling] + summaries:
        lines.append(f"| {s['model']} | " + " | ".join(pct(s["per_component"][c]) for c in COMPONENTS) + " |")
    lines += ["", "## Disagreements with the reference (solo, first run)", ""]
    for s in [ceiling] + summaries:
        lines += [f"### {s['model']}", ""]
        if not s["misses"]:
            lines += ["None.", ""]
        for m in s["misses"][: 25]:
            if m["error"]:
                lines.append(f"- `{m['id']}`: ERROR {m['error']}")
                continue
            diffs = "; ".join(f"{c}: got `{m['pred'][c]}` vs ref `{m['ref'][c]}`" for c in m["wrong"])
            lines.append(f"- `{m['id']}`: {diffs}")
        lines.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def revalidate(row: dict, item: dict) -> dict:
    """Re-run the current validator + escalation rule on a saved raw model output (no API call)."""
    if "raw_plan" not in row:
        return row
    req = VisualizeRequest(**item["request"])
    plan, adj = finalize_plan(QueryPlan(**row["raw_plan"]), req)
    return {**row, "plan": plan.model_dump(exclude_none=True), "adjustments": adj,
            "escalate": escalation_reason_from(row["confidence"], row["ambiguities"], adj)}


def load_raw(path: Path, items: list[dict]) -> tuple[list[dict], list[dict], dict[str, list[list[dict]]]]:
    by_id = {it["id"]: it for it in items}
    passes: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["id"] in by_id:
            passes.setdefault(r.pop("pass"), []).append(revalidate(r, by_id[r["id"]]))
    cands: dict[str, list[list[dict]]] = {}
    for tag, rows in passes.items():
        if "#" in tag:
            cands.setdefault(tag.split("#")[0], []).append(rows)
    return passes["reference"], passes["second_pass"], cands


async def main(args: argparse.Namespace) -> None:
    items = json.loads((ROOT / "evals" / "queries.json").read_text())
    if args.limit:
        items = items[: args.limit]

    if args.rescore:
        out_dir = Path(args.rescore).resolve()
        ref_rows, esc_rows, cand_rows = load_raw(out_dir / "raw.jsonl", items)
        print(f"rescoring {out_dir} with the current validator / scorer (no API calls)")
    else:
        out_dir = ROOT / "evals" / "results" / datetime.now().strftime("%Y%m%d-%H%M")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{len(items)} queries -> {out_dir.relative_to(ROOT)}")
        ref_rows = await run_pass(items, args.reference, args.ref_effort, args.concurrency, f"{args.reference} (ref)")
        esc_rows = await run_pass(items, args.reference, args.ref_effort, args.concurrency, f"{args.reference} (2nd)")
        cand_rows = {}
        for m in args.candidates:
            cand_rows[m] = [await run_pass(items, m, args.effort, args.concurrency, f"{m} #{i + 1}")
                            for i in range(args.repeats)]
        write_raw(out_dir, ref_rows, esc_rows, cand_rows)

    finish(out_dir, items, ref_rows, esc_rows, cand_rows, args)


def write_raw(out_dir: Path, ref_rows: list[dict], esc_rows: list[dict], cand_rows: dict) -> None:
    with (out_dir / "raw.jsonl").open("w") as fh:
        for tag, rows in [("reference", ref_rows), ("second_pass", esc_rows)] + [
                (f"{m}#{i + 1}", r) for m, passes in cand_rows.items() for i, r in enumerate(passes)]:
            for r in rows:
                fh.write(json.dumps({"pass": tag, **r}, default=str) + "\n")


def finish(out_dir: Path, items: list[dict], ref_rows: list[dict], esc_rows: list[dict], cand_rows: dict,
           args: argparse.Namespace) -> None:
    ref = {r["id"]: r for r in ref_rows}
    esc = {r["id"]: r for r in esc_rows}
    ceiling = summarize(args.reference, [esc_rows], ref, None)
    summaries = [summarize(m, passes, ref, esc) for m, passes in cand_rows.items()]
    pick, why = choose(summaries)
    (out_dir / "summary.json").write_text(json.dumps({"ceiling": ceiling, "candidates": summaries,
                                                      "pick": pick["model"], "why": why}, indent=2, default=str))
    report = write_report(out_dir, args.reference, ceiling, summaries, pick, why, len(items), args)
    print(f"\nceiling {args.reference}: exact {pct(ceiling['exact'])}  ${ceiling['cost_per_query']:.5f}/q")
    for s in summaries:
        print(f"{s['model']:<14} exact {pct(s['exact'])}  cascade {pct(s['cascade_exact'])}  "
              f"esc {pct(s['escalation_rate'])}  ${s['cost_per_query']:.5f}/q solo  "
              f"${s['cascade_cost_per_query']:.5f}/q cascade")
    print(f"\nPICK: {pick['model']} -- {why}\nreport: {report.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=settings.escalation_model)
    ap.add_argument("--ref-effort", default=settings.escalation_effort)
    ap.add_argument("--candidates", nargs="+", default=["gpt-5-mini", "gpt-5.4-nano", "gpt-4.1-mini"])
    ap.add_argument("--effort", default=settings.planner_effort)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries (debugging)")
    ap.add_argument("--rescore", metavar="RESULTS_DIR", help="re-score saved raw outputs without calling the API")
    asyncio.run(main(ap.parse_args()))
