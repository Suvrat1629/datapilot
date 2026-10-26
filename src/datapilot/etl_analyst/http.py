"""Outbound HTTP for the ETL agent, locked down to an allow-list."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

import httpx

from datapilot.config import Settings
from datapilot.etl_analyst.store import EtlToolError

Scalar = str | int | float | bool


def allowed_hosts(settings: Settings) -> set[str]:
    return {h.strip().lower() for h in settings.etl_allowed_hosts.split(",") if h.strip()}


def check_url(url: str, hosts: set[str]) -> str:
    """Require https and an exact, allow-listed hostname. Returns the host."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise EtlToolError("only https URLs are allowed")
    if parts.username or parts.password:
        raise EtlToolError("URLs with credentials are not allowed")
    host = (parts.hostname or "").lower()
    if not hosts:
        raise EtlToolError("no hosts are allow-listed for ETL (ETL_ALLOWED_HOSTS is empty)")
    if host not in hosts:
        raise EtlToolError(
            f"host {host!r} is not allow-listed. Allowed hosts: {', '.join(sorted(hosts))}"
        )
    if parts.port not in (None, 443):
        raise EtlToolError("non-standard ports are not allowed")
    return host


def _clean_params(params: dict[str, Any] | None) -> dict[str, Scalar]:
    out: dict[str, Scalar] = {}
    for key, value in (params or {}).items():
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            out[key] = value
        else:
            raise EtlToolError(f"parameter {key!r} must be a string, number or boolean")
    return out


def fetch_json(
    url: str, params: dict[str, Any] | None, settings: Settings, client: httpx.Client | None = None
) -> Any:
    """GET `url` and return parsed JSON. No redirects, hard size cap, bounded time."""
    check_url(url, allowed_hosts(settings))
    cap = settings.etl_max_response_bytes
    own = client is None
    client = client or httpx.Client()
    try:
        with client.stream(
            "GET",
            url,
            params=_clean_params(params),
            follow_redirects=False,
            timeout=settings.etl_fetch_timeout_s,
        ) as resp:
            if 300 <= resp.status_code < 400:
                raise EtlToolError(f"HTTP {resp.status_code} redirect; redirects are not followed")
            body = bytearray()
            for chunk in resp.iter_bytes():
                body.extend(chunk)
                if len(body) > cap:
                    raise EtlToolError(f"response larger than {cap:,} bytes; narrow the request")
            if resp.status_code >= 400:
                snippet = body[:300].decode("utf-8", "replace")
                raise EtlToolError(f"HTTP {resp.status_code} from the API: {snippet}")
    except httpx.TimeoutException:
        raise EtlToolError(f"request timed out after {settings.etl_fetch_timeout_s:.0f}s") from None
    except httpx.HTTPError as e:
        raise EtlToolError(f"request failed: {type(e).__name__}") from None
    finally:
        if own:
            client.close()
    try:
        return json.loads(bytes(body))
    except ValueError:
        raise EtlToolError("the API did not return JSON") from None
