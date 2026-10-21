"""Runtime configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# USD per 1M tokens (input, cached input, output). Source: OpenAI pricing as of 2026-10;
# update here if prices change -- used for cost reporting in responses and evals.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}


def cost_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    fresh = max(input_tokens - cached_tokens, 0)
    return (fresh * p[0] + cached_tokens * p[1] + output_tokens * p[2]) / 1_000_000


def is_reasoning_model(model: str) -> bool:
    """gpt-5 family accepts `reasoning.effort`; gpt-4.x models do not."""
    return model.startswith(("gpt-5", "o"))


@dataclass(frozen=True)
class Settings:
    ctgov_base_url: str = os.getenv("CTGOV_BASE_URL", "https://clinicaltrials.gov/api/v2")
    ctgov_timeout_s: float = float(os.getenv("CTGOV_TIMEOUT_S", "30"))
    ctgov_page_size: int = 1000  # API maximum
    cache_ttl_s: int = int(os.getenv("CTGOV_CACHE_TTL_S", "900"))
    cache_max_items: int = int(os.getenv("CTGOV_CACHE_MAX_ITEMS", "256"))

    # LLM planner cascade: small model first, escalate to the large model when the
    # small model's plan fails validation, is low-confidence, or the call errors.
    llm_enabled: bool = _bool("CTVIZ_LLM_ENABLED", True)
    planner_model: str = os.getenv("CTVIZ_PLANNER_MODEL", "gpt-5-mini")
    planner_effort: str = os.getenv("CTVIZ_PLANNER_EFFORT", "low")
    escalation_model: str = os.getenv("CTVIZ_ESCALATION_MODEL", "gpt-5.4")
    escalation_effort: str = os.getenv("CTVIZ_ESCALATION_EFFORT", "medium")
    escalation_enabled: bool = _bool("CTVIZ_ESCALATION_ENABLED", True)
    llm_timeout_s: float = float(os.getenv("CTVIZ_LLM_TIMEOUT_S", "60"))

    # Public-deployment guardrails (0 = unlimited). After the daily LLM budget is used up,
    # requests fall back to the free rule-based planner instead of failing.
    daily_llm_limit: int = int(os.getenv("CTVIZ_DAILY_LLM_LIMIT", "0"))
    rate_limit_per_min: int = int(os.getenv("CTVIZ_RATE_LIMIT_PER_MIN", "0"))

    default_max_studies: int = int(os.getenv("CTVIZ_DEFAULT_MAX_STUDIES", "5000"))
    hard_max_studies: int = int(os.getenv("CTVIZ_HARD_MAX_STUDIES", "10000"))
    known_models: tuple[str, ...] = field(default=tuple(PRICES))


settings = Settings()


def llm_available() -> bool:
    """True when an OpenAI credential is configured and the LLM planner is enabled."""
    return settings.llm_enabled and bool(os.getenv("OPENAI_API_KEY"))
