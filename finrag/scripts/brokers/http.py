"""
Shared HTTP layer for broker adapters: JSON requests with retry/backoff on rate
limits and transient failures, mapped to the typed errors in base.py.

The session is injectable, so tests run fully offline with recorded responses.
Credentials live only in request headers; they are redacted from every message
this module produces.
"""
from __future__ import annotations

import time
from typing import Callable

from .base import BrokerUnavailable, MalformedResponse, RateLimited

# classify(status_code, parsed_json_or_None) -> an exception to raise, "retry", or None (= success)
Classifier = Callable[[int, object], object]


class BrokerHTTP:
    def __init__(self, base_url: str, headers: dict[str, str], source: str, classify: Classifier,
                 session=None, secrets: tuple[str, ...] = (), sleep=time.sleep,
                 max_retries: int = 3, base_delay: float = 1.0, timeout: float = 20.0):
        if session is None:
            import requests
            session = requests.Session()
        self.base_url = base_url.rstrip("/")
        self._headers = headers
        self.source = source
        self.classify = classify
        self.session = session
        self._secrets = tuple(s for s in secrets if s)
        self.sleep = sleep
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout

    def redact(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, "***")
        return text

    def request(self, method: str, path: str, *, params: dict | None = None, json_body=None):
        """Parsed JSON of a successful response; retries rate limits / transient errors with backoff."""
        url = f"{self.base_url}{path}"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(method, url, params=params, json=json_body,
                                            headers=self._headers, timeout=self.timeout)
            except Exception as e:  # connection refused, DNS, timeout — never echo the request
                if type(e).__name__ == "NetworkBlocked":  # the sync's network guard: a bug, not an outage
                    raise
                last = BrokerUnavailable(f"{self.source}: network error ({type(e).__name__})", self.source)
                self._backoff(attempt)
                continue
            try:
                body = resp.json() if resp.content else None
            except ValueError:
                body = None
            verdict = self.classify(resp.status_code, body)
            if verdict is None:
                if body is None:
                    raise MalformedResponse(f"{self.source}: HTTP {resp.status_code} with a non-JSON body", self.source)
                return body
            if verdict == "retry" or isinstance(verdict, (RateLimited, BrokerUnavailable)):
                last = verdict if isinstance(verdict, Exception) else RateLimited(
                    f"{self.source}: rate limited (HTTP {resp.status_code})", self.source)
                self._backoff(attempt)
                continue
            if isinstance(verdict, Exception):
                raise type(verdict)(self.redact(str(verdict)), self.source)
            raise MalformedResponse(f"{self.source}: unexpected classification {verdict!r}", self.source)
        assert last is not None
        raise type(last)(self.redact(str(last)) + f" — gave up after {self.max_retries + 1} attempts", self.source)

    def _backoff(self, attempt: int) -> None:
        if attempt < self.max_retries:
            self.sleep(self.base_delay * (2 ** attempt))  # 1 s, 2 s, 4 s …
