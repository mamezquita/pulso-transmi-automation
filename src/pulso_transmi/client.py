from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterator

import httpx
import pandas as pd


DEFAULT_BASE_URL = "https://pulso-transmi.72-60-245-2.sslip.io"


class PulsoTransmiError(RuntimeError):
    """Raised when the Pulso TransMi API cannot fulfill a request."""


class PulsoTransmiClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved_url = base_url or os.getenv("PULSO_API_URL", DEFAULT_BASE_URL)
        resolved_key = api_key or os.getenv("PULSO_API_KEY")
        headers = {"User-Agent": "pulso-transmi-python/0.1.0"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"
        self._client = httpx.Client(
            base_url=resolved_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def __enter__(self) -> "PulsoTransmiClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        try:
            response = self._client.get(path, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise PulsoTransmiError(f"GET {path} failed: {exc}") from exc

    def meta(self) -> dict[str, Any]:
        return self._get("/v1/meta").json()

    def stations(self) -> pd.DataFrame:
        payload = self._get("/v1/stations").json()
        frame = pd.DataFrame(payload["data"])
        if not frame.empty:
            frame["station_id"] = frame["station_id"].astype("string")
        return frame

    def observations_page(
        self,
        *,
        station_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        params = {
            "station_id": station_id,
            "start": start,
            "end": end,
            "cursor": cursor,
            "limit": limit,
        }
        return self._get("/v1/observations", params={key: value for key, value in params.items() if value is not None}).json()

    def context_page(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        params = {"start": start, "end": end, "cursor": cursor, "limit": limit}
        return self._get("/v1/context", params={key: value for key, value in params.items() if value is not None}).json()

    def _all_pages(self, endpoint: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        cursor = None
        seen: set[str] = set()
        while True:
            page_params = {**params, "cursor": cursor}
            payload = self._get(endpoint, params={key: value for key, value in page_params.items() if value is not None}).json()
            yield from payload["data"]
            cursor = payload.get("next_cursor")
            if cursor is None:
                break
            if cursor in seen:
                raise PulsoTransmiError("API returned a repeated cursor")
            seen.add(cursor)

    def observations_dataframe(
        self,
        *,
        station_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        page_size: int = 5000,
    ) -> pd.DataFrame:
        rows = self._all_pages(
            "/v1/observations",
            {"station_id": station_id, "start": start, "end": end, "limit": page_size},
        )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
            frame["station_id"] = frame["station_id"].astype("string")
        return frame

    def context_dataframe(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        page_size: int = 5000,
    ) -> pd.DataFrame:
        rows = self._all_pages(
            "/v1/context", {"start": start, "end": end, "limit": page_size}
        )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        return frame

    def download(self, filename: str, destination: str | Path) -> Path:
        allowed = {"stations.csv", "observations.csv", "context.csv", "metadata.json"}
        if filename not in allowed:
            raise ValueError(f"unsupported filename: {filename}")
        response = self._get(f"/v1/downloads/{filename}")
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)

        if filename != "metadata.json":
            expected = self.meta()["dataset"]["files"][filename]["sha256"]
            actual = hashlib.sha256(response.content).hexdigest()
            if actual != expected:
                path.unlink(missing_ok=True)
                raise PulsoTransmiError(f"checksum mismatch for {filename}")
        return path
