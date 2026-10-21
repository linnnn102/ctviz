"""The QueryPlan: the single contract between "understanding" and "execution".

Both planners (LLM and rules) emit a QueryPlan; the validator repairs it;
the engine executes it deterministically. The LLM never sees or produces
data values -- it only fills this small, enum-constrained structure.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.vocab import (
    ChartType,
    Dimension,
    Intent,
    InterventionType,
    Phase,
    SponsorClass,
    Status,
    StudyType,
)


class SearchFilters(BaseModel):
    """Filters that map 1:1 onto ClinicalTrials.gov API query parameters."""

    model_config = ConfigDict(extra="forbid")

    condition: str | None = Field(None, description="Disease/condition text (query.cond).")
    intervention: str | None = Field(None, description="Drug/intervention text (query.intr).")
    sponsor: str | None = Field(None, description="Sponsor/collaborator text (query.spons).")
    term: str | None = Field(None, description="Free-text keyword when the entity type is unclear (query.term).")
    country: str | None = Field(None, description="Country with a trial site, full English name.")
    phases: list[Phase] | None = None
    statuses: list[Status] | None = None
    study_type: StudyType | None = None
    intervention_types: list[InterventionType] | None = None
    sponsor_classes: list[SponsorClass] | None = None
    start_year_min: int | None = None
    start_year_max: int | None = None

    def merged(self, override: SearchFilters | None) -> SearchFilters:
        """Return a copy where non-null fields of `override` replace ours."""
        if override is None:
            return self.model_copy()
        data = self.model_dump()
        data.update({k: v for k, v in override.model_dump().items() if v is not None})
        return SearchFilters(**data)

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())


class Cohort(BaseModel):
    """One side of a comparison. Its filters are layered on top of plan.filters."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., description="Short legend label, e.g. 'Pembrolizumab'.")
    filters: SearchFilters


class NetworkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Dimension = Field(..., description="Node type A (entity dimension).")
    target: Dimension = Field(..., description="Node type B. Same as source => co-occurrence network.")


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    group_by: Dimension | None = Field(
        None,
        description="distribution: category dim; time_trend: temporal dim; histogram: numeric dim.",
    )
    series_by: Dimension | None = Field(
        None, description="Optional secondary categorical split (e.g. phase within each year)."
    )
    x: Dimension | None = Field(None, description="scatter: numeric x dimension.")
    y: Dimension | None = Field(None, description="scatter: numeric y dimension.")
    network: NetworkSpec | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    cohorts: list[Cohort] = Field(
        default_factory=list, description="Two or more cohorts => comparison along series."
    )
    chart_type: ChartType | None = None
    top_n: int | None = None
    rationale: str = Field("", description="One or two sentences explaining the interpretation.")
