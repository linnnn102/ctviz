"""Translate SearchFilters into ClinicalTrials.gov v2 /studies query params.

Free-text entities use the API's dedicated search areas (query.cond /
query.intr / query.spons / query.term) so we inherit its synonym expansion
(e.g. "Keytruda" -> pembrolizumab). Enumerated filters use the Essie
`filter.advanced` syntax, verified against the live API.
"""

from __future__ import annotations

from app.plan import SearchFilters


def _quote(value: str) -> str:
    return '"' + value.replace('"', "") + '"'


def to_api_params(f: SearchFilters) -> dict[str, str]:
    params: dict[str, str] = {}
    if f.condition:
        params["query.cond"] = f.condition
    if f.intervention:
        params["query.intr"] = f.intervention
    if f.sponsor:
        params["query.spons"] = f.sponsor
    if f.term:
        params["query.term"] = f.term
    if f.statuses:
        params["filter.overallStatus"] = ",".join(sorted(set(f.statuses)))

    adv: list[str] = []
    if f.phases:
        adv.append("AREA[Phase](" + " OR ".join(sorted(set(f.phases))) + ")")
    if f.study_type:
        adv.append(f"AREA[StudyType]{f.study_type}")
    if f.intervention_types:
        adv.append("AREA[InterventionType](" + " OR ".join(sorted(set(f.intervention_types))) + ")")
    if f.sponsor_classes:
        adv.append("AREA[LeadSponsorClass](" + " OR ".join(sorted(set(f.sponsor_classes))) + ")")
    if f.country:
        adv.append(f"AREA[LocationCountry]{_quote(f.country)}")
    if f.start_year_min or f.start_year_max:
        lo = f"{f.start_year_min}-01-01" if f.start_year_min else "MIN"
        hi = f"{f.start_year_max}-12-31" if f.start_year_max else "MAX"
        adv.append(f"AREA[StartDate]RANGE[{lo},{hi}]")
    if adv:
        params["filter.advanced"] = " AND ".join(adv)
    return params
