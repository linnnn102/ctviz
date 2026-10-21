"""Async ClinicalTrials.gov v2 client with paging, retries and a TTL cache."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings

log = logging.getLogger(__name__)


class CTGovError(RuntimeError):
    """Upstream API failure that survived retries (or a 4xx we caused)."""

    def __init__(self, message: str, status: int | None = None, params: dict | None = None):
        super().__init__(message)
        self.status = status
        self.params = params


@dataclass
class FetchResult:
    studies: list[dict[str, Any]]
    total_count: int
    truncated: bool
    params: dict[str, str]


class _TTLCache:
    def __init__(self, ttl_s: int, max_items: int = 256):
        self.ttl_s, self.max_items = ttl_s, max_items
        self._data: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any | None:
        hit = self._data.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl_s:
            return hit[1]
        self._data.pop(key, None)
        return None

    def put(self, key: Any, value: Any) -> None:
        if len(self._data) >= self.max_items:
            self._data.pop(next(iter(self._data)))
        self._data[key] = (time.monotonic(), value)


class CTGovClient:
    RETRY_STATUSES = {429, 500, 502, 503, 504}

    def __init__(self, http: httpx.AsyncClient | None = None, max_retries: int = 3):
        self._http = http or httpx.AsyncClient(
            base_url=settings.ctgov_base_url,
            timeout=settings.ctgov_timeout_s,
            headers={"User-Agent": "ctviz/0.1 (clinical-trials visualization agent)"},
        )
        self._max_retries = max_retries
        self._cache = _TTLCache(settings.cache_ttl_s, settings.cache_max_items)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        delay = 0.5
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._http.get(path, params=params)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise CTGovError(f"ClinicalTrials.gov unreachable: {exc}", params=params) from exc
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in self.RETRY_STATUSES or attempt == self._max_retries:
                    raise CTGovError(
                        f"ClinicalTrials.gov returned {resp.status_code}: {resp.text[:300]}",
                        status=resp.status_code,
                        params=params,
                    )
            log.warning("ctgov retry %s for %s", attempt + 1, params)
            await asyncio.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    async def search(self, params: dict[str, str], fields: list[str], max_studies: int) -> FetchResult:
        """Fetch up to `max_studies` matching studies, projecting only `fields`."""
        key = (tuple(sorted(params.items())), tuple(sorted(fields)), max_studies)
        if (cached := self._cache.get(key)) is not None:
            return cached

        studies: list[dict[str, Any]] = []
        total = 0
        token: str | None = None
        while True:
            page_size = min(settings.ctgov_page_size, max_studies - len(studies))
            q = {**params, "fields": ",".join(fields), "pageSize": str(page_size), "format": "json"}
            if token:
                q["pageToken"] = token
            else:
                q["countTotal"] = "true"
            body = await self._get("/studies", q)
            if not token:
                total = int(body.get("totalCount", 0))
            studies.extend(body.get("studies", []))
            token = body.get("nextPageToken")
            if not token or len(studies) >= max_studies:
                break

        result = FetchResult(studies=studies, total_count=total, truncated=len(studies) < total, params=params)
        self._cache.put(key, result)
        return result

    async def count(self, params: dict[str, str]) -> int:
        body = await self._get("/studies", {**params, "countTotal": "true", "pageSize": "1", "fields": "NCTId"})
        return int(body.get("totalCount", 0))

    async def data_timestamp(self) -> str | None:
        if (cached := self._cache.get("version")) is not None:
            return cached
        try:
            body = await self._get("/version", {})
        except CTGovError:
            return None
        ts = body.get("dataTimestamp")
        self._cache.put("version", ts)
        return ts
