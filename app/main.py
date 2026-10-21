"""HTTP API.

POST /visualize  question -> visualization spec + metadata (the main endpoint)
POST /plan       question -> validated plan only (debugging / transparency)
GET  /dimensions catalog of dimensions the engine supports
GET  /health     liveness + whether the LLM planner is configured
GET  /           demo UI (static/index.html)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app import dimensions
from app.analysis.engine import Engine
from app.config import llm_available, settings
from app.ctgov.client import CTGovClient, CTGovError
from app.planner import llm_budget, make_plan
from app.schemas import ErrorResponse, VisualizeRequest, VisualizeResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = CTGovClient()
    app.state.engine = Engine(client)
    yield
    await client.aclose()


app = FastAPI(
    title="ClinicalTrials.gov Query-to-Visualization Agent",
    version="0.1.0",
    description="Converts clinical-trial questions into structured visualization specs backed by "
                "ClinicalTrials.gov API v2 data, with per-datum citations.",
    lifespan=lifespan,
)


class RateLimiter:
    """Sliding one-minute window per client IP (in-memory; fine for a single-instance demo)."""

    def __init__(self) -> None:
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, ip: str, limit: int) -> bool:
        now = time.monotonic()
        q = self.hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        if len(self.hits) > 10_000:  # drop idle clients
            for k in [k for k, v in self.hits.items() if not v or now - v[-1] > 60]:
                del self.hits[k]
        return True


rate_limiter = RateLimiter()
LIMITED_PATHS = {"/visualize", "/plan"}


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    limit = settings.rate_limit_per_min
    if limit and request.method == "POST" and request.url.path in LIMITED_PATHS:
        # behind Render's proxy the client is the first X-Forwarded-For entry
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
        if not rate_limiter.allow(ip, limit):
            return JSONResponse(status_code=429, headers={"Retry-After": "60"}, content=ErrorResponse(
                error="rate_limited", detail=f"max {limit} requests per minute").model_dump())
    return await call_next(request)


@app.exception_handler(CTGovError)
async def _ctgov_error(_: Request, exc: CTGovError) -> JSONResponse:
    return JSONResponse(status_code=502, content=ErrorResponse(
        error="upstream_error", detail={"message": str(exc), "status": exc.status, "params": exc.params}
    ).model_dump())


@app.post("/visualize", response_model=VisualizeResponse, responses={422: {"model": ErrorResponse},
                                                                     502: {"model": ErrorResponse}})
async def visualize(req: VisualizeRequest, request: Request) -> VisualizeResponse:
    return await request.app.state.engine.run(req)


@app.post("/plan")
async def plan(req: VisualizeRequest) -> dict:
    pr = await make_plan(req)
    return {"planner": pr.planner, "model": pr.model, "escalated": pr.escalated,
            "escalation_reason": pr.escalation_reason, "plan": pr.plan.model_dump(exclude_none=True),
            "adjustments": pr.adjustments, "warnings": pr.warnings, "llm_trace": pr.trace}


@app.get("/dimensions")
async def list_dimensions() -> list[dict]:
    return [{"name": d.name, "title": d.title, "kind": d.kind, "multi_valued": d.multi,
             "network_capable": d.entity, "unit": d.unit, "api_fields": list(d.api_fields)}
            for d in dimensions.REGISTRY.values()]


@app.get("/health")
async def health() -> dict:
    on = llm_available()
    return {"status": "ok", "llm_planner": on,
            "llm_requests_today": llm_budget.used, "daily_llm_limit": settings.daily_llm_limit or None,
            "planner_model": settings.planner_model if on else None,
            "escalation_model": settings.escalation_model if on and settings.escalation_enabled else None}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
