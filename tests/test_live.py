"""Live checks against the real ClinicalTrials.gov API.  Run with: uv run pytest -m live

The key test independently re-fetches cited trials and verifies every excerpt
is verbatim at the cited field path -- i.e. citations are not just plausible,
they are mechanically checkable.
"""

import asyncio
import re

import httpx
import pytest

from app.analysis.engine import Engine
from app.ctgov.client import CTGovClient
from app.schemas import VisualizeRequest

pytestmark = pytest.mark.live


def resolve(study: dict, path: str):
    cur = study
    for part in path.split("."):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        cur = cur[m.group(1)]
        if m.group(2) is not None:
            cur = cur[int(m.group(2))]
    return cur


def as_text(v) -> str:
    if isinstance(v, list):
        return ", ".join(v)
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


async def fetch_study(http: httpx.AsyncClient, nct: str) -> dict:
    """GET one study, backing off on the API's per-IP rate limit (~50 req/min)."""
    for attempt in range(5):
        r = await http.get(f"/studies/{nct}", params={"format": "json"})
        if r.status_code == 200 and r.content:
            return r.json()
        await asyncio.sleep(2 * (attempt + 1))
    raise AssertionError(f"could not fetch {nct}: HTTP {r.status_code}")


QUERIES = [
    {"query": "How are melanoma trials distributed across phases?"},
    {"query": "Which countries have the most recruiting trials for lung cancer?"},
    {"query": "Show a network of sponsors ↔ drugs for melanoma trials"},
    {"query": "How has the number of trials for this drug changed over time?", "drug_name": "Pembrolizumab"},
]


@pytest.mark.parametrize("kw", QUERIES)
async def test_citations_are_verbatim(kw):
    client = CTGovClient()
    try:
        resp = await Engine(client).run(VisualizeRequest(**kw, use_llm=False, max_citations_per_datum=2))
    finally:
        await client.aclose()
    data = resp.visualization.data
    items = data["nodes"] + data["edges"] if isinstance(data, dict) else data
    cites = [c for it in items for c in it["citations"]][:12]
    assert cites
    async with httpx.AsyncClient(base_url="https://clinicaltrials.gov/api/v2", timeout=30) as http:
        for c in cites:
            study = await fetch_study(http, c["nct_id"])
            for ex in c["excerpts"]:
                if ex["field"].endswith("locations") or ex["field"].endswith("armGroups"):
                    continue  # count-derived excerpts ("12 locations")
                assert as_text(resolve(study, ex["field"])) == ex["text"], (c["nct_id"], ex)


async def test_bar_totals_match_api_counts():
    """Single-valued distribution must match per-category API counts exactly when untruncated."""
    client = CTGovClient()
    try:
        resp = await Engine(client).run(VisualizeRequest(query="glioblastoma trials by sponsor class", use_llm=False))
        assert not resp.meta.coverage[0].truncated
        industry = next(r for r in resp.visualization.data if r["sponsor_class"] == "Industry")
        n = await client.count({"query.cond": "glioblastoma", "filter.advanced": "AREA[LeadSponsorClass]INDUSTRY"})
        assert industry["trial_count"] == n
    finally:
        await client.aclose()
