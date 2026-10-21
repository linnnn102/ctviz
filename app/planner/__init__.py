"""Planning stage: question -> validated QueryPlan, via a model cascade.

    small planner model ──► validator ──ok──────────────────────────► plan
          │ error / low confidence / severe validator repairs
          ▼
    escalation model (gpt-5.4) ──► validator ──► plan
          │ error
          ▼
    small model's plan if it had one, else the rule-based planner

Every attempt is recorded in PlanResult.trace (model, latency, tokens, cost,
why it was accepted or escalated) and surfaced in the response metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal

from app.config import llm_available, settings
from app.plan import QueryPlan
from app.planner.llm import LLMCall, PlannerError, plan_with_llm
from app.planner.rules import plan_with_rules
from app.planner.validate import finalize_plan, severe_adjustments
from app.schemas import VisualizeRequest

log = logging.getLogger(__name__)


@dataclass
class PlanResult:
    plan: QueryPlan
    planner: Literal["llm", "rules"]
    adjustments: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model: str | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


def _trace_entry(call: LLMCall | None, model: str, outcome: str, detail: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"model": model, "outcome": outcome}
    if detail:
        entry["detail"] = detail
    if call:
        cost = call.cost_usd
        entry.update(latency_ms=call.latency_ms, input_tokens=call.input_tokens, output_tokens=call.output_tokens,
                     reasoning_tokens=call.reasoning_tokens, confidence=call.output.confidence,
                     cost_usd=round(cost, 6) if cost is not None else None)
        if call.output.ambiguities:
            entry["ambiguities"] = call.output.ambiguities
    return entry


def escalation_reason(call: LLMCall, adjustments: list[str]) -> str | None:
    """Why a small-model plan should not be trusted as-is (None = accept)."""
    return escalation_reason_from(call.output.confidence, call.output.ambiguities, adjustments)


def escalation_reason_from(confidence: str, ambiguities: list[str], adjustments: list[str]) -> str | None:
    if confidence == "low":
        doubts = "; ".join(ambiguities) or "no details"
        return f"low confidence ({doubts})"
    if sev := severe_adjustments(adjustments):
        return "validator had to repair the plan: " + "; ".join(sev)
    return None


class DailyBudget:
    """Counts LLM-planned requests per UTC day; in-memory, so it resets on restart (per process)."""

    def __init__(self) -> None:
        self.day: date | None = None
        self.used = 0

    def take(self, limit: int) -> bool:
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day, self.used = today, 0
        if limit and self.used >= limit:
            return False
        self.used += 1
        return True


llm_budget = DailyBudget()


async def make_plan(req: VisualizeRequest) -> PlanResult:
    if not (req.use_llm and llm_available()):
        plan, adj = finalize_plan(plan_with_rules(req), req)
        return PlanResult(plan=plan, planner="rules", adjustments=adj)
    if not llm_budget.take(settings.daily_llm_limit):
        plan, adj = finalize_plan(plan_with_rules(req), req)
        return PlanResult(plan=plan, planner="rules", adjustments=adj,
                          warnings=[f"daily LLM budget ({settings.daily_llm_limit} requests) reached; "
                                    "used rule-based planner"])

    trace: list[dict[str, Any]] = []
    warnings: list[str] = []
    small = settings.planner_model
    candidate: tuple[QueryPlan, list[str], LLMCall] | None = None
    reason: str | None

    # tier 1: small model
    try:
        call = await plan_with_llm(req, small, settings.planner_effort)
        plan, adj = finalize_plan(call.output.plan, req)
        candidate = (plan, adj, call)
        reason = escalation_reason(call, adj)
    except PlannerError as exc:
        call, reason = None, f"error: {exc}"

    if reason is None:
        trace.append(_trace_entry(call, small, "accepted"))
        return PlanResult(plan=plan, planner="llm", adjustments=adj, model=small, trace=trace)
    trace.append(_trace_entry(call, small, "escalated", reason))

    # tier 2: large model
    big = settings.escalation_model
    if settings.escalation_enabled and big != small:
        try:
            call2 = await plan_with_llm(req, big, settings.escalation_effort)
            plan2, adj2 = finalize_plan(call2.output.plan, req)
            trace.append(_trace_entry(call2, big, "accepted"))
            return PlanResult(plan=plan2, planner="llm", adjustments=adj2, model=big, escalated=True,
                              escalation_reason=reason, trace=trace)
        except PlannerError as exc:
            trace.append(_trace_entry(None, big, "failed", str(exc)))
            warnings.append(f"escalation model {big} failed ({exc})")

    # tier 3: best effort
    if candidate is not None:
        plan, adj, _ = candidate
        warnings.append(f"used {small} plan despite: {reason}")
        return PlanResult(plan=plan, planner="llm", adjustments=adj, model=small, escalated=False,
                          escalation_reason=reason, warnings=warnings, trace=trace)
    warnings.append("LLM planners unavailable; used rule-based planner")
    plan, adj = finalize_plan(plan_with_rules(req), req)
    return PlanResult(plan=plan, planner="rules", adjustments=adj, warnings=warnings, trace=trace)
