"""Run the example queries against the live API and save request+response JSON.

    uv run python scripts/run_examples.py            # all examples
    uv run python scripts/run_examples.py --no-llm   # force the rule-based planner

Outputs go to examples/ (primary 5) and examples/extra/ (more chart types).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.analysis.engine import Engine  # noqa: E402
from app.ctgov.client import CTGovClient  # noqa: E402
from app.schemas import VisualizeRequest  # noqa: E402

PRIMARY = {
    "01_time_trend_pembrolizumab": {"query": "How has the number of trials for this drug changed over time?",
                                    "drug_name": "Pembrolizumab"},
    "02_compare_phases_pembro_vs_nivo": {"query": "Compare phases for trials involving Pembrolizumab vs Nivolumab"},
    "03_countries_recruiting_lung_cancer": {"query": "Which countries have the most recruiting trials for lung cancer?"},
    "04_network_sponsor_drug_melanoma": {"query": "Show a network of sponsors ↔ drugs for melanoma trials",
                                         "top_n": 12},
    "05_network_drug_cooccurrence_pancreatic": {
        "query": "Which drugs frequently co-occur in combination studies for pancreatic cancer?", "top_n": 12},
}
EXTRA = {
    "histogram_enrollment_phase3_diabetes": {"query": "What is the distribution of enrollment sizes for phase 3 "
                                                      "diabetes trials?"},
    "scatter_enrollment_vs_duration_alzheimers": {"query": "Plot enrollment vs duration for Alzheimer's disease "
                                                           "trials", "max_studies": 300},
    "grouped_sponsor_class_asthma_vs_copd": {"query": "Compare sponsor categories across two conditions.",
                                             "compare": ["asthma", "COPD"], "compare_on": "condition"},
    "time_trend_by_phase_melanoma_since_2015": {"query": "How many melanoma trials started each year since 2015 "
                                                         "by phase?"},
    "metric_recruiting_semaglutide": {"query": "How many recruiting trials are there for semaglutide?"},
    "table_recruiting_glioblastoma_germany": {"query": "List recruiting trials for glioblastoma in Germany"},
}


async def main(no_llm: bool, cites: int) -> None:
    client = CTGovClient()
    engine = Engine(client)
    try:
        for folder, examples in ((ROOT / "examples", PRIMARY), (ROOT / "examples" / "extra", EXTRA)):
            folder.mkdir(parents=True, exist_ok=True)
            for name, body in examples.items():
                body = {"max_citations_per_datum": cites, **body}
                if no_llm:
                    body["use_llm"] = False
                req = VisualizeRequest(**body)
                resp = await engine.run(req)
                out = {"request": req.model_dump(exclude_defaults=True) | {"query": req.query},
                       "response": resp.model_dump(mode="json")}
                (folder / f"{name}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
                v = resp.visualization
                print(f"{name}: {v.type} | {v.title} | planner={resp.meta.interpretation.planner} | "
                      f"{resp.meta.elapsed_ms} ms")
                await asyncio.sleep(1)  # be polite to the public API
    finally:
        await client.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--citations", type=int, default=2, help="max citations per datum in saved outputs")
    args = ap.parse_args()
    asyncio.run(main(args.no_llm, args.citations))
