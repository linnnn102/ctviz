"""Closed vocabularies shared by the request schema, the planner and the engine.

Keeping these as Literals means the LLM's structured output is constrained to
exactly the values the engine knows how to execute.
"""

from __future__ import annotations

from typing import Literal, get_args

Phase = Literal["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]

Status = Literal[
    "NOT_YET_RECRUITING",
    "RECRUITING",
    "ENROLLING_BY_INVITATION",
    "ACTIVE_NOT_RECRUITING",
    "SUSPENDED",
    "TERMINATED",
    "COMPLETED",
    "WITHDRAWN",
    "UNKNOWN",
]

StudyType = Literal["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"]

InterventionType = Literal[
    "DRUG",
    "BIOLOGICAL",
    "DEVICE",
    "PROCEDURE",
    "RADIATION",
    "BEHAVIORAL",
    "GENETIC",
    "DIETARY_SUPPLEMENT",
    "COMBINATION_PRODUCT",
    "DIAGNOSTIC_TEST",
    "OTHER",
]

SponsorClass = Literal["INDUSTRY", "NIH", "FED", "OTHER_GOV", "INDIV", "NETWORK", "AMBIG", "OTHER", "UNKNOWN"]

ChartType = Literal[
    "bar_chart",
    "grouped_bar_chart",
    "stacked_bar_chart",
    "time_series",
    "histogram",
    "scatter_plot",
    "network_graph",
    "metric",
    "table",
]

Intent = Literal[
    "count",          # single number -> metric
    "distribution",   # counts per category -> bar
    "time_trend",     # counts per year -> time_series
    "histogram",      # distribution of a numeric trial attribute
    "scatter",        # numeric vs numeric per trial
    "network",        # entity co-occurrence graph
    "list",           # list of matching trials -> table
]

# Every dimension the engine can extract from a study record.
Dimension = Literal[
    # categorical
    "phase",
    "status",
    "study_type",
    "sponsor",
    "sponsor_class",
    "intervention_type",
    "drug",          # DRUG/BIOLOGICAL interventions, normalized names
    "intervention",  # any intervention, normalized names
    "condition",
    "country",
    "primary_purpose",
    "allocation",
    "has_results",
    "investigator",
    "site",
    # temporal
    "start_year",
    "completion_year",
    "first_posted_year",
    # numeric
    "enrollment",
    "duration_months",
    "site_count",
    "arm_count",
]

PHASE_LABELS: dict[str, str] = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not Applicable",
}

STATUS_LABELS: dict[str, str] = {s: s.replace("_", " ").title() for s in get_args(Status)}

ALL_PHASES = list(get_args(Phase))
ALL_DIMENSIONS = list(get_args(Dimension))
