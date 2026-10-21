"""LLM planner: natural language -> QueryPlan via OpenAI structured outputs.

The model's only job is *interpretation*. It returns a QueryPlan constrained
by a strict JSON schema (enums for intents, dimensions, phases, statuses...).
It never sees trial data and never produces numbers that end up in a chart,
so there is nothing for it to hallucinate except the plan itself -- and the
plan is validated and grounded afterwards (see validate.py).

The same function serves both tiers of the cascade (small planner model and
the gpt-5.4 escalation model); see planner/__init__.py.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Literal

import openai
from pydantic import BaseModel, ConfigDict, Field

from app import dimensions
from app.config import cost_usd, is_reasoning_model, settings
from app.plan import QueryPlan
from app.schemas import VisualizeRequest

log = logging.getLogger(__name__)


class PlannerError(RuntimeError):
    pass


class LLMPlan(BaseModel):
    """What the model returns: the plan plus a self-assessment used for escalation."""

    model_config = ConfigDict(extra="forbid")

    plan: QueryPlan
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="low if the question is ambiguous or you had to guess the entity type or intent."
    )
    ambiguities: list[str] = Field(..., description="Short notes on anything you were unsure about; empty if none.")


@dataclass
class LLMCall:
    output: LLMPlan
    model: str
    latency_ms: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int

    @property
    def cost_usd(self) -> float | None:
        return cost_usd(self.model, self.input_tokens, self.cached_tokens, self.output_tokens)


SYSTEM_PROMPT = f"""You translate questions about clinical trials into a QueryPlan that a deterministic engine
executes against the ClinicalTrials.gov v2 API. You never answer the question yourself and never invent data.

## Intents
- count: a single number ("how many trials ...?") with no grouping.
- distribution: counts per category of a categorical dimension (bar chart). Use for "most common", "distributed
  across", "which countries/sponsors have the most", "breakdown by".
- time_trend: counts per year (group_by must be a temporal dimension, usually start_year).
- histogram: distribution of a numeric attribute (enrollment, duration_months, site_count, arm_count).
- scatter: one numeric dimension against another, one point per trial (x and y). "Y vs X" means y=Y, x=X.
- network: relationships between entities. network.source/target are entity dimensions; use the same dimension
  twice for co-occurrence (e.g. drug<->drug = drugs tested together in the same trial).
- list: the user wants to see individual trials (a table).

## Dimensions
{dimensions.catalog_for_prompt()}

Use "drug" (drug/biological interventions) rather than "intervention" unless the user asks about interventions
in general (devices, procedures, behavioral ...). Use "sponsor_class" for sponsor categories/types (industry vs
NIH vs academic) and "sponsor" for named sponsors. series_by is an optional second categorical split
("per year by phase" -> group_by=start_year, series_by=phase).

## Filters
filters apply to every cohort. Put a drug in `intervention`, a disease in `condition`, a company/institution in
`sponsor`, and use `term` only when you cannot tell which it is. `country` must be a full English country name as
used by ClinicalTrials.gov (e.g. "United States", "United Kingdom", "Korea, Republic of"). Map "recruiting" to
statuses ["RECRUITING"]; "active" trials to ["RECRUITING","ACTIVE_NOT_RECRUITING","ENROLLING_BY_INVITATION"].
A specific phase ("phase 3 trials") is a filter (phases), while "across phases" is a grouping (group_by=phase).
"NIH-sponsored" / "industry-sponsored" are sponsor_classes filters. Years refer to trial start date unless stated
otherwise ("since 2015" -> start_year_min 2015; "in the last 5 years" -> start_year_min = current year - 5).

## Comparisons
"A vs B", "compare A and B", "across two conditions X and Y": create one cohort per compared entity with only
the differing filter in cohort.filters, and keep shared filters in `filters`. Cohort labels are short legend
names. Do not create cohorts when nothing is being compared. "X vs Y" between two numeric attributes is a scatter,
not a comparison.

## Hard rules
- Only use entities that literally appear in the question or in the structured fields. Never add a drug, condition,
  sponsor, country, phase, status, study type, intervention type, sponsor class or year the user did not ask to
  restrict by. Correct obvious misspellings only; do not translate brand names to generic names.
- Grouping or networking by a dimension is not a filter on it: "which drugs co-occur" uses the drug dimension and
  needs no intervention_types filter; set intervention_types only for explicit restrictions such as "device trials".
- If the question refers to "this drug"/"this condition", the structured fields supply it; leave that filter null.
- Leave fields null when not needed; top_n and chart_type null unless the user asks for them. Keep `rationale` to one
  sentence.
- Set confidence to "low" when the question is ambiguous, off-topic, or you had to guess; list the doubts in
  `ambiguities`.
"""


def _user_message(req: VisualizeRequest) -> str:
    structured = req.model_dump(
        exclude={"query", "top_n", "max_studies", "max_citations_per_datum", "include_nct_ids", "use_llm"},
        exclude_none=True,
        exclude_defaults=True,
    )
    return json.dumps({"today": date.today().isoformat(), "question": req.query, "structured_fields": structured},
                      indent=2)


_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(timeout=settings.llm_timeout_s, max_retries=2)
    return _client


async def plan_with_llm(req: VisualizeRequest, model: str, effort: str | None = None) -> LLMCall:
    kwargs: dict = {}
    if is_reasoning_model(model) and effort:
        kwargs["reasoning"] = {"effort": effort}
    t0 = time.perf_counter()
    try:
        response = await _get_client().responses.parse(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=_user_message(req),
            text_format=LLMPlan,
            max_output_tokens=16000,
            store=False,
            **kwargs,
        )
    except openai.RateLimitError as exc:
        raise PlannerError(f"{model} rate limited") from exc
    except openai.APITimeoutError as exc:
        raise PlannerError(f"{model} timed out") from exc
    except openai.APIStatusError as exc:
        raise PlannerError(f"{model} API error {exc.status_code}: {exc.message}") from exc
    except openai.APIConnectionError as exc:
        raise PlannerError(f"{model} unreachable: {exc}") from exc
    except openai.OpenAIError as exc:  # parse/validation failures, missing credentials, ...
        raise PlannerError(f"{model} client error: {exc}") from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", "unknown")
        raise PlannerError(f"{model} output incomplete ({reason})")
    for item in response.output:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                raise PlannerError(f"{model} refused: {part.refusal}")
    parsed = response.output_parsed
    if parsed is None:
        raise PlannerError(f"{model} returned no parseable plan")

    u = response.usage
    call = LLMCall(
        output=parsed,
        model=model,
        latency_ms=latency_ms,
        input_tokens=u.input_tokens if u else 0,
        cached_tokens=(u.input_tokens_details.cached_tokens or 0) if u and u.input_tokens_details else 0,
        output_tokens=u.output_tokens if u else 0,
        reasoning_tokens=(u.output_tokens_details.reasoning_tokens or 0) if u and u.output_tokens_details else 0,
    )
    log.info("%s plan (%s, %d ms, $%.5f): %s", model, parsed.confidence, latency_ms, call.cost_usd or 0,
             parsed.plan.model_dump_json(exclude_none=True))
    return call
