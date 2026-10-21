import pytest

from app import normalize as nz
from app.ctgov.query import to_api_params
from app.plan import SearchFilters


@pytest.mark.parametrize("phases,label", [
    (["PHASE3"], "Phase 3"),
    (["PHASE1", "PHASE2"], "Phase 1/Phase 2"),
    (["NA"], "Not Applicable"),
    (["EARLY_PHASE1"], "Early Phase 1"),
    ([], "Unknown"),
    (None, "Unknown"),
])
def test_phase_label(phases, label):
    assert nz.phase_label(phases) == label
    assert label in nz.PHASE_ORDER


@pytest.mark.parametrize("name,other,mesh,aliases,key", [
    ("Pembrolizumab 200 mg IV", None, ["pembrolizumab"], None, "pembrolizumab"),
    ("Pembrolizumab 100 mg/4 mL (25 mg/ml) Injection", None, [], None, "pembrolizumab"),
    ("MK-3475", ["Keytruda", "Pembrolizumab"], [], {"pembrolizumab": "pembrolizumab"}, "pembrolizumab"),
    ("Matching Placebo tablets", None, [], None, nz.PLACEBO_KEY),
    ("Nab-paclitaxel", None, ["Paclitaxel", "Albumin-Bound Paclitaxel"], None, "nab-paclitaxel"),
    ("Lenvatinib Capsules", None, ["lenvatinib"], None, "lenvatinib"),
])
def test_intervention_key(name, other, mesh, aliases, key):
    assert nz.intervention_key(name, other, mesh, aliases) == key


def test_mesh_requires_whole_word():
    # "insulin" must not be matched inside "insulinoma"-like tokens
    assert nz.intervention_key("Proinsulinx", None, ["insulin"]) == "proinsulinx"


@pytest.mark.parametrize("a,b", [
    ("Merck Sharp & Dohme LLC", "Merck Sharp & Dohme Corp."),
    ("Acme Pharma Inc.", "Acme Pharma, Inc"),
    ("Jiangsu Hengrui Medicine Co., Ltd.", "Jiangsu HengRui Medicine Co Ltd"),
])
def test_sponsor_key_folds_corporate_suffixes(a, b):
    assert nz.sponsor_key(a) == nz.sponsor_key(b)


def test_sponsor_key_keeps_inner_words():
    assert nz.sponsor_key("Co-operative Oncology Group") != ""
    assert "operative" in nz.sponsor_key("Co-operative Oncology Group")


def test_dates():
    assert nz.year_of("2016-03") == 2016
    assert nz.year_of(None) is None
    assert nz.year_of("n/a") is None
    assert nz.months_between("2016-03", "2019-03") == 36.0
    assert nz.months_between("2020-01", "2019-01") is None


def test_to_api_params():
    f = SearchFilters(condition="melanoma", intervention="pembrolizumab", phases=["PHASE3", "PHASE2"],
                      statuses=["RECRUITING"], country="United States", start_year_min=2015, study_type="INTERVENTIONAL")
    p = to_api_params(f)
    assert p["query.cond"] == "melanoma"
    assert p["query.intr"] == "pembrolizumab"
    assert p["filter.overallStatus"] == "RECRUITING"
    adv = p["filter.advanced"]
    assert "AREA[Phase](PHASE2 OR PHASE3)" in adv
    assert 'AREA[LocationCountry]"United States"' in adv
    assert "AREA[StartDate]RANGE[2015-01-01,MAX]" in adv
    assert "AREA[StudyType]INTERVENTIONAL" in adv


def test_to_api_params_empty():
    assert to_api_params(SearchFilters()) == {}
