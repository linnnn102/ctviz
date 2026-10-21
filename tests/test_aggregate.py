from app.analysis import aggregate as agg
from app.dimensions import ExtractContext


def opts(**kw):
    return agg.Opts(**{"cap": 5, "top_n": 15, **kw})


def by(rows, key):
    return {r[key]: r for r in rows}


def test_phase_distribution_counts_and_order(studies):
    res = agg.distribution([agg.Cohort("all", studies, 4)], "phase", opts())
    rows = by(res.data, "phase")
    assert rows["Phase 3"]["trial_count"] == 2
    assert rows["Phase 1/Phase 2"]["trial_count"] == 1
    assert rows["Unknown"]["trial_count"] == 1
    # fixed ordinal order, not count order
    assert [r["phase"] for r in res.data] == ["Phase 1/Phase 2", "Phase 3", "Unknown"]
    assert sum(r["trial_count"] for r in res.data) == 4  # single-valued: partitions trials


def test_citations_are_exact_api_values(studies):
    res = agg.distribution([agg.Cohort("all", studies, 4)], "phase", opts())
    cite = by(res.data, "phase")["Phase 1/Phase 2"]["citations"][0]
    assert cite["nct_id"] == "NCT00000002"
    assert cite["url"] == "https://clinicaltrials.gov/study/NCT00000002"
    assert cite["excerpts"] == [{"field": "protocolSection.designModule.phases", "text": "PHASE1, PHASE2"}]


def test_multivalued_country_counts_each_trial_once(studies):
    res = agg.distribution([agg.Cohort("all", studies, 4)], "country", opts())
    rows = by(res.data, "country")
    assert rows["United States"]["trial_count"] == 1  # listed twice in one trial
    assert rows["Germany"]["trial_count"] == 2
    exc = rows["Germany"]["citations"][0]["excerpts"][0]
    assert exc["field"].startswith("protocolSection.contactsLocationsModule.locations[")
    assert exc["text"] == "Germany"


def test_sponsor_variants_grouped(studies):
    res = agg.distribution([agg.Cohort("all", studies, 4)], "sponsor", opts())
    acme = [r for r in res.data if r["sponsor"].startswith("Acme")]
    assert len(acme) == 1 and acme[0]["trial_count"] == 3


def test_drug_alias_folds_code_names(studies):
    ctx = ExtractContext(intervention_aliases={"pembrolizumab": "pembrolizumab"})
    res = agg.distribution([agg.Cohort("all", studies, 4)], "drug", opts(ctx=ctx))
    rows = by(res.data, "drug")
    assert rows["Pembrolizumab"]["trial_count"] == 2  # "Pembrolizumab 200 mg IV" + "MK-3475"
    assert "Placebo" not in rows  # comparator excluded from drug dimension
    assert rows["Ipilimumab"]["trial_count"] == 2


def test_citation_cap_and_ids(studies):
    res = agg.distribution([agg.Cohort("all", studies, 4)], "sponsor_class", opts(cap=1, include_ids=True))
    ind = by(res.data, "sponsor_class")["Industry"]
    assert ind["trial_count"] == 3 and ind["citation_count"] == 3
    assert len(ind["citations"]) == 1
    assert len(ind["nct_ids"]) == 3


def test_top_n_truncation_note(studies):
    res = agg.distribution([agg.Cohort("all", studies, 4)], "intervention", opts(top_n=3))
    assert len(res.data) == 3
    assert any("top 3" in n for n in res.notes)


def test_grouped_distribution_has_explicit_zeros(studies):
    a = agg.Cohort("A", studies[:2], 2)
    b = agg.Cohort("B", studies[2:], 2)
    res = agg.distribution([a, b], "phase", opts())
    cells = {(r["phase"], r["cohort"]): r["trial_count"] for r in res.data}
    assert cells[("Phase 1/Phase 2", "B")] == 0
    assert cells[("Phase 3", "A")] == 1 and cells[("Phase 3", "B")] == 1
    assert res.series_order == ["A", "B"]
    assert len(res.data) == len(res.category_order) * 2


def test_time_trend_fills_gaps_and_respects_range(studies):
    res = agg.time_trend([agg.Cohort("all", studies, 4)], "start_year", opts(), None, None, None)
    years = [r["year"] for r in res.data]
    assert years == [2016, 2017, 2018, 2019, 2020]
    assert by(res.data, "year")[2017]["trial_count"] == 0
    assert by(res.data, "year")[2018]["trial_count"] == 2
    clipped = agg.time_trend([agg.Cohort("all", studies, 4)], "start_year", opts(), None, 2018, None)
    assert [r["year"] for r in clipped.data] == [2018, 2019, 2020]


def test_histogram_partitions_trials(studies):
    res = agg.histogram([agg.Cohort("all", studies, 4)], "enrollment", opts())
    assert sum(r["trial_count"] for r in res.data) == 3  # one trial lacks enrollment
    assert res.extra["stats"]["n"] == 3
    for r in res.data:
        if not r["overflow"]:
            assert r["bin_end"] > r["bin_start"]
    assert any("lack enrollment" in n for n in res.notes)


def test_drug_cooccurrence_network(studies):
    ctx = ExtractContext(intervention_aliases={"pembrolizumab": "pembrolizumab"})
    res = agg.network([agg.Cohort("all", studies, 4)], "drug", "drug", opts(ctx=ctx))
    nodes = {n["id"]: n for n in res.data["nodes"]}
    edges = {(e["source"], e["target"]): e for e in res.data["edges"]}
    assert ("drug:ipilimumab", "drug:pembrolizumab") in edges
    assert ("drug:ipilimumab", "drug:nivolumab") in edges
    e = edges[("drug:ipilimumab", "drug:pembrolizumab")]
    assert e["weight"] == 1
    fields = [x["field"] for x in e["citations"][0]["excerpts"]]
    assert len(fields) == 2  # both endpoints cited from the same trial
    assert nodes["drug:pembrolizumab"]["is_query_entity"] is True
    assert "drug:placebo" not in nodes


def test_bipartite_network(studies):
    res = agg.network([agg.Cohort("all", studies, 4)], "sponsor", "drug", opts())
    types = {n["type"] for n in res.data["nodes"]}
    assert types == {"sponsor", "drug"}
    for e in res.data["edges"]:
        assert e["source"].startswith("sponsor:") and e["target"].startswith("drug:")


def test_network_dedups_overlapping_cohorts(studies):
    res = agg.network([agg.Cohort("a", studies, 4), agg.Cohort("b", studies, 4)], "drug", "drug", opts())
    assert res.extra["trials_in_graph"] == 4
    assert all(e["weight"] == 1 for e in res.data["edges"])


def test_scatter_one_point_per_trial(studies):
    res = agg.scatter([agg.Cohort("all", studies, 4)], "duration_months", "enrollment", opts(), None)
    assert {r["nct_id"] for r in res.data} == {"NCT00000001", "NCT00000002"}
    assert all(len(r["citations"][0]["excerpts"]) == 3 for r in res.data)  # start, completion, enrollment


def test_counts_use_api_total(studies):
    res = agg.counts([agg.Cohort("x", studies[:2], 1234)], opts(cap=2))
    assert res.data[0]["trial_count"] == 1234
    assert len(res.data[0]["citations"]) == 2
