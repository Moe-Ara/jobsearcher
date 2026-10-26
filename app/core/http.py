"""Small polite HTTP client shared by adapters: per-host delay, size cap, retries, URL safety check."""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.settings import get_settings

log = logging.getLogger(__name__)

MAX_BYTES = 8 * 1024 * 1024
# Job APIs return every posting with its full description in one response (Greenhouse ?content=true for an employer
# with 400+ openings is ~20 MB). JSON is parsed, never rendered — a larger cap is safe there.
JSON_MAX_BYTES = 64 * 1024 * 1024


class FetchError(RuntimeError):
    pass


def assert_public_http_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts resolving to private/loopback addresses (for user-pasted URLs)."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise FetchError("Only http(s) URLs are supported")
    try:
        infos = socket.getaddrinfo(parts.hostname, None)
    except socket.gaierror as exc:
        raise FetchError(f"Cannot resolve host {parts.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise FetchError("URL points to a private/internal address")


class HttpClient:
    def __init__(self) -> None:
        s = get_settings()
        self._client = httpx.Client(
            timeout=s.http_timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": s.http_user_agent,
                "Accept-Language": "en-US,en;q=0.8,de;q=0.6",
            },
            transport=httpx.HTTPTransport(retries=2),
        )
        self._min_interval = s.http_min_interval_seconds
        self._last_hit: dict[str, float] = {}
        self._lock = threading.Lock()

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        with self._lock:
            wait = self._last_hit.get(host, 0) + self._min_interval - time.monotonic()
            self._last_hit[host] = time.monotonic() + max(wait, 0)
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, *, params: dict[str, Any] | None = None, accept: str | None = None,
            max_bytes: int = MAX_BYTES, headers: dict[str, str] | None = None) -> httpx.Response:
        self._throttle(url)
        headers = {**({"Accept": accept} if accept else {}), **(headers or {})} or None
        try:
            with self._client.stream("GET", url, params=params, headers=headers) as resp:
                chunks, size = [], 0
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise FetchError(f"Response too large from {url}")
                    chunks.append(chunk)
                resp._content = b"".join(chunks)  # noqa: SLF001 - materialise streamed body
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__} fetching {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise FetchError(f"HTTP {resp.status_code} from {url}")
        return resp

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        resp = self.get(url, params=params, accept="application/json", max_bytes=JSON_MAX_BYTES)
        try:
            return resp.json()
        except ValueError as exc:
            raise FetchError(f"Invalid JSON from {url}") from exc

    def get_json_with_headers(self, url: str, *, params: dict[str, Any] | None = None,
                              headers: dict[str, str] | None = None) -> Any:
        """JSON API that needs a client id header (Bundesagentur Jobsuche)."""
        resp = self.get(url, params=params, accept="application/json", max_bytes=JSON_MAX_BYTES, headers=headers)
        try:
            return resp.json()
        except ValueError as exc:
            raise FetchError(f"Invalid JSON from {url}") from exc

    def post_json(self, url: str, payload: dict[str, Any]) -> Any:
        """JSON POST for read-only search APIs (Workday CXS)."""
        self._throttle(url)
        try:
            resp = self._client.post(url, json=payload, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__} posting to {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise FetchError(f"HTTP {resp.status_code} from {url}")
        if len(resp.content) > JSON_MAX_BYTES:
            raise FetchError(f"Response too large from {url}")
        try:
            return resp.json()
        except ValueError as exc:
            raise FetchError(f"Invalid JSON from {url}") from exc

    def get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        return self.get(url, params=params, accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8").text

    def close(self) -> None:
        self._client.close()


_shared: HttpClient | None = None
_shared_lock = threading.Lock()


def get_http() -> HttpClient:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = HttpClient()
        return _shared
