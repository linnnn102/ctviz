import pytest

from app.plan import Cohort, NetworkSpec, QueryPlan, SearchFilters
from app.planner.rules import plan_with_rules
from app.planner.validate import finalize_plan, is_grounded, severe_adjustments
from app.schemas import VisualizeRequest


def rules(**kw):
    req = VisualizeRequest(**kw)
    plan, adj = finalize_plan(plan_with_rules(req), req)
    return plan, adj


# The appendix query classes of the assignment, plus a few extra
@pytest.mark.parametrize("kw,expect", [
    ({"query": "How has the number of trials for this drug changed over time?", "drug_name": "Pembrolizumab"},
     {"intent": "time_trend", "group_by": "start_year", "intervention": "Pembrolizumab"}),
    ({"query": "How has the number of trials for pembrolizumab changed per year since 2015?"},
     {"intent": "time_trend", "intervention": "pembrolizumab", "start_year_min": 2015}),
    ({"query": "How many trials started each year for Alzheimer's disease?"},
     {"intent": "time_trend", "condition": "Alzheimer's disease"}),
    ({"query": "How are melanoma trials distributed across phases?"},
     {"intent": "distribution", "group_by": "phase", "condition": "melanoma"}),
    ({"query": "What are the most common intervention types for breast cancer trials?"},
     {"intent": "distribution", "group_by": "intervention_type", "condition": "breast cancer"}),
    ({"query": "Which countries have the most recruiting trials for lung cancer?"},
     {"intent": "distribution", "group_by": "country", "condition": "lung cancer", "statuses": ["RECRUITING"]}),
    ({"query": "What is the distribution of enrollment sizes for phase 3 diabetes trials?"},
     {"intent": "histogram", "group_by": "enrollment", "condition": "diabetes", "phases": ["PHASE3"]}),
    ({"query": "How many recruiting trials are there for semaglutide?"},
     {"intent": "count", "intervention": "semaglutide"}),
    ({"query": "List recruiting trials for glioblastoma in Germany"},
     {"intent": "list", "condition": "glioblastoma", "country": "Germany"}),
])
def test_rule_planner_examples(kw, expect):
    plan, _ = rules(**kw)
    for k, v in expect.items():
        actual = getattr(plan, k) if hasattr(plan, k) and k not in SearchFilters.model_fields else getattr(plan.filters, k)
        assert actual == v, (k, actual, plan)


def test_rule_planner_comparison():
    plan, _ = rules(query="Compare phases for trials involving Pembrolizumab vs Nivolumab.")
    assert plan.intent == "distribution" and plan.group_by == "phase"
    assert [c.label for c in plan.cohorts] == ["Pembrolizumab", "Nivolumab"]
    assert plan.cohorts[0].filters.intervention == "Pembrolizumab"


def test_rule_planner_networks():
    plan, _ = rules(query="Show a network of sponsors ↔ drugs for melanoma trials.")
    assert plan.network == NetworkSpec(source="sponsor", target="drug")
    plan, _ = rules(query="Which drugs frequently co-occur in combination studies (drug ↔ drug network)?")
    assert plan.network == NetworkSpec(source="drug", target="drug")
    plan, _ = rules(query="Which drugs frequently co-occur in combination studies for pancreatic cancer?")
    assert plan.network == NetworkSpec(source="drug", target="drug")
    assert plan.filters.condition == "pancreatic cancer"


def test_rule_planner_scatter_is_not_a_comparison():
    plan, _ = rules(query="Plot enrollment vs duration for Alzheimer's disease trials")
    assert plan.intent == "scatter" and plan.cohorts == []
    assert (plan.y, plan.x) == ("enrollment", "duration_months")


def test_compare_field_builds_cohorts():
    plan, adj = rules(query="Compare sponsor categories across two conditions.", compare=["asthma", "COPD"],
                      compare_on="condition")
    assert plan.group_by == "sponsor_class"
    assert [c.filters.condition for c in plan.cohorts] == ["asthma", "COPD"]


# ------------------------------------------------------------- validator ---


def test_grounding_drops_hallucinated_entities():
    req = VisualizeRequest(query="How are melanoma trials distributed across phases?")
    llm_plan = QueryPlan(intent="distribution", group_by="phase",
                         filters=SearchFilters(condition="melanoma", intervention="nivolumab", country="France"))
    plan, adj = finalize_plan(llm_plan, req)
    assert plan.filters.condition == "melanoma"
    assert plan.filters.intervention is None and plan.filters.country is None
    assert any("ungrounded" in a and "nivolumab" in a for a in adj)


def test_grounding_tolerates_typos_and_aliases():
    assert is_grounded("pembrolizumab", ["trials for pembrolizumb per year"])
    assert is_grounded("United States", ["recruiting trials in the US"])
    assert not is_grounded("nivolumab", ["trials for pembrolizumab"])
    # planners may add generic words to an entity the user named
    assert is_grounded("Parkinson's disease", ["Breakdown of status for Parkinson's trials"])
    assert is_grounded("Alzheimer disease", ["How long do Alzheimer's trials typically last?"])
    # ...but not rename it or add a specific word
    assert not is_grounded("pembrolizumab", ["Which conditions does Keytruda get tested in most?"])
    assert not is_grounded("lung cancer", ["trials for breast cancer"])
    assert not is_grounded("disease", ["how many trials?"])


def test_redundant_design_filters_removed():
    req = VisualizeRequest(query="Which drugs frequently co-occur in combination studies for pancreatic cancer?")
    p = QueryPlan(intent="network", network=NetworkSpec(source="drug", target="drug"),
                  filters=SearchFilters(condition="pancreatic cancer", term="combination", intervention_types=["DRUG"]))
    plan, adj = finalize_plan(p, req)
    assert plan.filters.term is None and plan.filters.intervention_types is None
    assert plan.filters.condition == "pancreatic cancer"
    assert not severe_adjustments(adj)  # normalization, not a planner failure


def test_explicit_intervention_type_kept():
    req = VisualizeRequest(query="Which drugs are used in these trials?", intervention_type=["DRUG"])
    plan, _ = finalize_plan(QueryPlan(intent="distribution", group_by="drug"), req)
    assert plan.filters.intervention_types == ["DRUG"]
    # ...and a device restriction is meaningful even with a drug grouping
    p = QueryPlan(intent="distribution", group_by="drug", filters=SearchFilters(intervention_types=["DEVICE"]))
    plan, _ = finalize_plan(p, VisualizeRequest(query="drugs used alongside device trials"))
    assert plan.filters.intervention_types == ["DEVICE"]


def test_structured_fields_override_plan():
    req = VisualizeRequest(query="trials by phase for keytruda", drug_name="Pembrolizumab", trial_phase=["PHASE3"])
    plan, adj = finalize_plan(QueryPlan(intent="distribution", group_by="phase",
                                        filters=SearchFilters(intervention="keytruda")), req)
    assert plan.filters.intervention == "Pembrolizumab"
    assert plan.filters.phases == ["PHASE3"]
    assert any("overrides" in a for a in adj)


def test_single_cohort_merged_and_dropped_cohorts():
    req = VisualizeRequest(query="phases for pembrolizumab")
    p = QueryPlan(intent="distribution", group_by="phase",
                  cohorts=[Cohort(label="P", filters=SearchFilters(intervention="pembrolizumab")),
                           Cohort(label="X", filters=SearchFilters(intervention="madeupumab"))])
    plan, adj = finalize_plan(p, req)
    assert plan.cohorts == [] and plan.filters.intervention == "pembrolizumab"


@pytest.mark.parametrize("plan,check", [
    (QueryPlan(intent="distribution", group_by="start_year"), lambda p: p.intent == "time_trend"),
    (QueryPlan(intent="distribution", group_by="enrollment"), lambda p: p.intent == "histogram"),
    (QueryPlan(intent="distribution"), lambda p: p.group_by == "phase"),
    (QueryPlan(intent="time_trend", group_by="phase"), lambda p: (p.group_by, p.series_by) == ("start_year", "phase")),
    (QueryPlan(intent="histogram", group_by="country"), lambda p: p.group_by == "enrollment"),
    (QueryPlan(intent="scatter", x="phase", y="phase"), lambda p: p.x != p.y and p.x and p.y),
    (QueryPlan(intent="network", network=NetworkSpec(source="phase", target="enrollment")),
     lambda p: p.network.source == "drug" and p.network.target == "drug"),
])
def test_dimension_repairs(plan, check):
    req = VisualizeRequest(query="something about trials")
    fixed, adj = finalize_plan(plan, req)
    assert check(fixed), fixed
    assert adj


def test_incompatible_chart_type_ignored():
    req = VisualizeRequest(query="melanoma trials by phase", chart_type="network_graph")
    plan, adj = finalize_plan(QueryPlan(intent="distribution", group_by="phase"), req)
    assert plan.chart_type is None
    assert any("incompatible" in a for a in adj)


def test_group_by_override_changes_intent():
    req = VisualizeRequest(query="melanoma trials", group_by="start_year")
    plan, _ = finalize_plan(QueryPlan(intent="distribution", group_by="phase"), req)
    assert plan.intent == "time_trend" and plan.group_by == "start_year"


def test_request_validation():
    with pytest.raises(ValueError):
        VisualizeRequest(query="x")
    with pytest.raises(ValueError):
        VisualizeRequest(query="valid question", start_year=2020, end_year=2010)
    with pytest.raises(ValueError):
        VisualizeRequest(query="valid question", unknown_field=1)
    with pytest.raises(ValueError):
        VisualizeRequest(query="valid question", trial_phase=["PHASE9"])
