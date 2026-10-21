"""Shared fixtures: hand-built study records in the exact shape of the v2 API
(projected with `fields=`), so aggregation can be tested offline and exactly."""

from __future__ import annotations

import pytest


def make_study(
    nct: str,
    *,
    phases=None,
    status="COMPLETED",
    start=None,
    completion=None,
    sponsor="Acme Pharma Inc.",
    sponsor_class="INDUSTRY",
    interventions=(),
    intervention_mesh=(),
    conditions=(),
    condition_mesh=(),
    countries=(),
    enrollment=None,
    title=None,
):
    ps: dict = {
        "identificationModule": {"nctId": nct, "briefTitle": title or f"Study {nct}"},
        "statusModule": {"overallStatus": status},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": sponsor, "class": sponsor_class}},
        "designModule": {},
        "armsInterventionsModule": {"interventions": [
            {"type": t, "name": n, **({"otherNames": list(o)} if o else {})} for t, n, *rest in interventions
            for o in [rest[0] if rest else None]
        ]},
        "conditionsModule": {"conditions": list(conditions)},
        "contactsLocationsModule": {"locations": [{"country": c} for c in countries]} if countries else {},
    }
    if phases is not None:
        ps["designModule"]["phases"] = phases
    if start:
        ps["statusModule"]["startDateStruct"] = {"date": start}
    if completion:
        ps["statusModule"]["completionDateStruct"] = {"date": completion}
    if enrollment is not None:
        ps["designModule"]["enrollmentInfo"] = {"count": enrollment}
    study = {"protocolSection": ps, "derivedSection": {}}
    if intervention_mesh:
        study["derivedSection"]["interventionBrowseModule"] = {"meshes": [{"term": t} for t in intervention_mesh]}
    if condition_mesh:
        study["derivedSection"]["conditionBrowseModule"] = {"meshes": [{"term": t} for t in condition_mesh]}
    return study


@pytest.fixture
def studies():
    return [
        make_study("NCT00000001", phases=["PHASE3"], start="2016-03", completion="2019-03",
                   interventions=[("DRUG", "Pembrolizumab 200 mg IV"), ("DRUG", "Placebo")],
                   intervention_mesh=["pembrolizumab"], condition_mesh=["Melanoma"],
                   countries=["United States", "United States", "Germany"], enrollment=400),
        make_study("NCT00000002", phases=["PHASE1", "PHASE2"], start="2018-01-15", completion="2020-01-15",
                   sponsor="Acme Pharma, Inc", interventions=[("BIOLOGICAL", "MK-3475", ["Pembrolizumab", "Keytruda"]),
                                                                ("DRUG", "Ipilimumab")],
                   intervention_mesh=["ipilimumab"], condition_mesh=["Melanoma", "Lung Neoplasms"],
                   countries=["Germany"], enrollment=60),
        make_study("NCT00000003", phases=["PHASE3"], start="2018-07", sponsor="National Cancer Institute (NCI)",
                   sponsor_class="NIH", interventions=[("DRUG", "Ipilimumab"), ("DRUG", "Nivolumab"),
                                                         ("RADIATION", "Radiation therapy")],
                   intervention_mesh=["ipilimumab", "nivolumab"], conditions=["Melanoma Stage IV"],
                   enrollment=1200),
        make_study("NCT00000004", phases=None, status="RECRUITING", start="2020-02",
                   interventions=[("BEHAVIORAL", "Exercise")], conditions=["Melanoma"], enrollment=None),
    ]
