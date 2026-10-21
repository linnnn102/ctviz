"""The planner-eval scorer must treat equivalent plans as equal and different plans as different."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from run_eval import compare  # noqa: E402


def plan(**kw):
    return {"intent": "distribution", "group_by": "phase", "filters": {}, "cohorts": [], **kw}


def test_identical_plans_match():
    p = plan(filters={"condition": "melanoma", "phases": ["PHASE3"]})
    assert all(compare(p, p).values())


def test_shared_filter_placement_is_equivalent():
    shared = plan(filters={"condition": "breast cancer"},
                  cohorts=[{"label": "CN", "filters": {"country": "China"}},
                           {"label": "US", "filters": {"country": "United States"}}])
    repeated = plan(cohorts=[{"label": "China", "filters": {"country": "China", "condition": "Breast Cancer"}},
                             {"label": "USA", "filters": {"country": "United States", "condition": "breast cancer"}}])
    assert all(compare(repeated, shared).values())


def test_network_direction_and_entity_spelling_tolerated():
    a = {"intent": "network", "network": {"source": "sponsor", "target": "drug"},
         "filters": {"condition": "Alzheimer's disease"}, "cohorts": []}
    b = {"intent": "network", "network": {"source": "drug", "target": "sponsor"},
         "filters": {"condition": "alzheimer"}, "cohorts": []}
    assert all(compare(a, b).values())


def test_real_differences_detected():
    ref = plan(filters={"condition": "melanoma"})
    assert not compare(plan(group_by="status", filters={"condition": "melanoma"}), ref)["dims"]
    assert not compare(plan(filters={"condition": "melanoma", "statuses": ["RECRUITING"]}), ref)["enums"]
    assert not compare(plan(filters={"intervention": "melanoma"}), ref)["entities"]  # wrong search area
    assert not compare(plan(filters={"condition": "melanoma", "start_year_min": 2015}), ref)["years"]
