"""Public request / response contract of the service.

These models ARE the documentation: FastAPI renders them at /docs and
/openapi.json, and README.md describes the same fields in prose.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.vocab import (
    ChartType,
    Dimension,
    InterventionType,
    Phase,
    SponsorClass,
    Status,
    StudyType,
)

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class VisualizeRequest(BaseModel):
    """Natural-language question plus optional structured hints.

    Structured fields always win over whatever the planner extracts from
    `query`; they are applied as hard filters to every cohort.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=3, max_length=1000, description="Natural-language question.")

    # --- structured filters (all optional) ---
    drug_name: str | None = Field(None, max_length=200, description="Intervention / drug search term.")
    condition: str | None = Field(None, max_length=200, description="Condition / disease search term.")
    sponsor: str | None = Field(None, max_length=200, description="Sponsor or collaborator search term.")
    country: str | None = Field(None, max_length=100, description='Country with at least one site, e.g. "Germany".')
    trial_phase: list[Phase] | None = Field(None, description="Restrict to these phases (OR).")
    status: list[Status] | None = Field(None, description="Restrict to these overall statuses (OR).")
    study_type: StudyType | None = None
    intervention_type: list[InterventionType] | None = None
    sponsor_class: list[SponsorClass] | None = None
    start_year: int | None = Field(None, ge=1900, le=2100, description="Trial start date >= Jan 1 of this year.")
    end_year: int | None = Field(None, ge=1900, le=2100, description="Trial start date <= Dec 31 of this year.")

    # --- output controls ---
    compare: list[str] | None = Field(
        None,
        min_length=2,
        max_length=6,
        description="Explicit cohorts to compare, e.g. ['pembrolizumab', 'nivolumab']. "
        "Interpreted along `compare_on`.",
    )
    compare_on: Literal["drug", "condition", "sponsor", "country"] = "drug"
    chart_type: ChartType | None = Field(None, description="Force a chart type (otherwise inferred).")
    group_by: Dimension | None = Field(None, description="Force the primary grouping dimension.")
    top_n: int = Field(15, ge=3, le=100, description="Max categories / nodes per type to display.")
    max_studies: int | None = Field(None, ge=50, le=10000, description="Max trial records analyzed per cohort.")
    max_citations_per_datum: int = Field(5, ge=0, le=50)
    include_nct_ids: bool = Field(False, description="Attach the full NCT id list to every datum.")
    use_llm: bool = Field(True, description="Set false to force the deterministic rule-based planner.")

    @field_validator("drug_name", "condition", "sponsor", "country", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _years_ordered(self) -> VisualizeRequest:
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        return self


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class Excerpt(BaseModel):
    field: str = Field(..., description="Dotted path in the ClinicalTrials.gov v2 study JSON.")
    text: str = Field(..., description="Exact value at that path, verbatim from the API response.")


class Citation(BaseModel):
    nct_id: str
    url: str
    title: str | None = Field(None, description="briefTitle of the trial (context only).")
    excerpts: list[Excerpt] = Field(..., description="Field values that justify counting this trial in the datum.")


class FieldEncoding(BaseModel):
    field: str = Field(..., description="Key in each data row (or node/edge object).")
    type: Literal["nominal", "ordinal", "quantitative", "temporal"]
    title: str | None = None
    unit: str | None = None
    sort: Literal["ascending", "descending"] | list[str] | None = Field(
        None, description="Explicit category order (list) or direction."
    )
    aggregate: Literal["count", "sum", "none"] | None = None


class NetworkEncoding(BaseModel):
    node_id: str = "id"
    node_label: str = "label"
    node_type: str = "type"
    node_size: str = "trial_count"
    edge_source: str = "source"
    edge_target: str = "target"
    edge_weight: str = "weight"


class Encoding(BaseModel):
    x: FieldEncoding | None = None
    y: FieldEncoding | None = None
    series: FieldEncoding | None = Field(None, description="Color / grouping channel (grouped bars, multi-line).")
    x2: FieldEncoding | None = Field(None, description="Bin end for histograms.")
    label: FieldEncoding | None = None
    value: FieldEncoding | None = Field(None, description="Single value for `metric`.")
    network: NetworkEncoding | None = None
    columns: list[FieldEncoding] | None = Field(None, description="Column order for `table`.")


class VisualizationOptions(BaseModel):
    orientation: Literal["vertical", "horizontal"] | None = None
    stacked: bool = False
    time_granularity: Literal["year"] | None = None
    x_scale: Literal["linear", "log"] | None = None
    y_scale: Literal["linear", "log"] | None = None
    layout: Literal["force"] | None = None
    bipartite: bool | None = None


class Visualization(BaseModel):
    type: ChartType
    title: str
    subtitle: str | None = None
    encoding: Encoding
    data: list[dict[str, Any]] | dict[str, list[dict[str, Any]]] = Field(
        ...,
        description="Rows for all charts; for network_graph an object {nodes: [...], edges: [...]}. "
        "Every row/node/edge carries `citations` (list[Citation]) and `citation_count`.",
    )
    options: VisualizationOptions = VisualizationOptions()


class CohortCoverage(BaseModel):
    label: str
    total_matching: int = Field(..., description="Trials matching the filters per the API's totalCount.")
    analyzed: int = Field(..., description="Trial records actually fetched and aggregated.")
    truncated: bool
    api_params: dict[str, str]


class Interpretation(BaseModel):
    planner: Literal["llm", "rules"]
    intent: str
    rationale: str | None = None
    model: str | None = Field(None, description="LLM whose plan was executed (null for the rule planner).")
    escalated: bool = Field(False, description="True if the small planner model was overruled by the large model.")
    escalation_reason: str | None = None
    plan: dict[str, Any] = Field(..., description="Validated plan that was executed (full transparency).")
    adjustments: list[str] = Field(default_factory=list, description="Corrections made by the validator.")
    llm_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="One entry per LLM call: model, outcome (accepted|escalated|failed), latency_ms, tokens, "
                    "cost_usd, confidence, detail.",
    )


class ResponseMeta(BaseModel):
    query: str
    source: str = "clinicaltrials.gov"
    source_api: str = "https://clinicaltrials.gov/api/v2/studies"
    filters: dict[str, Any]
    interpretation: Interpretation
    coverage: list[CohortCoverage]
    counting_rule: str
    sort: str | None = None
    units: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    generated_at: str
    elapsed_ms: int
    data_as_of: date | None = None


class VisualizeResponse(BaseModel):
    visualization: Visualization
    meta: ResponseMeta


class ErrorResponse(BaseModel):
    error: str
    detail: Any | None = None
