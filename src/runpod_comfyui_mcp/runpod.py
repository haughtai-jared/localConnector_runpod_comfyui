"""Minimal Runpod serverless client.

Only four calls are needed to run a ComfyUI workflow on a serverless endpoint:

``POST {base}/{endpoint}/run``            submit a job
``GET  {base}/{endpoint}/status/{id}``    poll it
``POST {base}/{endpoint}/cancel/{id}``    give up on it
``GET  {base}/{endpoint}/health``         check the endpoint is deployed

The client is deliberately synchronous: the MCP tools block until the image is
ready, and the SDK already runs tool bodies on a worker thread.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_JOB_TIMEOUT_S,
    DEFAULT_POLL_INTERVAL_S,
    mask_secret,
)

TERMINAL_OK = "COMPLETED"
TERMINAL_BAD = frozenset({"FAILED", "CANCELLED", "TIMED_OUT"})


class RunpodError(RuntimeError):
    """A Runpod request failed, or the endpoint rejected it."""


class RunpodJobError(RunpodError):
    """The job reached a terminal failure state."""

    def __init__(self, message: str, *, job_id: str | None = None, status: str = "", output: Any = None):
        super().__init__(message)
        self.job_id = job_id
        self.status = status
        self.output = output


class RunpodTimeout(RunpodError):
    """The job did not finish inside the configured timeout."""

    def __init__(self, message: str, *, job_id: str | None = None):
        super().__init__(message)
        self.job_id = job_id


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: str
    output: Any
    delay_time_ms: int | None = None
    execution_time_ms: int | None = None
    raw: dict | None = None


def _error_detail(response: httpx.Response) -> str:
    """Pull the most useful message out of a Runpod error body."""
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip().replace("\n", " ")
        return text[:300] or "<empty response>"

    if isinstance(body, dict):
        for key in ("error", "message", "detail"):
            value = body.get(key)
            if value:
                return str(value)[:300]
    return json.dumps(body)[:300]


def _numeric(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class RunpodClient:
    """Thin wrapper around the Runpod serverless REST API.

    Args:
        api_key: Runpod API key. Never logged, never sent anywhere but Runpod.
        endpoint_id: Serverless endpoint id (not a URL).
        base_url: API host; override for a proxy.
        http_client: Injected ``httpx.Client`` (tests use ``MockTransport``).
        timeout_s: Default wall-clock budget for a whole job.
        poll_interval_s: Delay between status polls.
        sleep / clock: Injected so tests do not actually wait.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint_id: str,
        base_url: str = DEFAULT_BASE_URL,
        http_client: httpx.Client | None = None,
        timeout_s: float = DEFAULT_JOB_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise RunpodError("Runpod API key is empty; cannot build a client.")
        if not endpoint_id:
            raise RunpodError("Runpod endpoint id is empty; cannot build a client.")

        self._api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = base_url.rstrip("/")
        self._base = f"{self.base_url}/{endpoint_id}"
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)
        )
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.max_attempts = max(1, int(max_attempts))
        self._sleep = sleep
        self._clock = clock

    # ── lifecycle ────────────────────────────────────────────────────────────
    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "RunpodClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── plumbing ─────────────────────────────────────────────────────────────
    @property
    def masked_api_key(self) -> str:
        return mask_secret(self._api_key)

    def _url(self, path: str = "") -> str:
        return f"{self._base}/{path.lstrip('/')}" if path else self._base

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(2.0 ** (attempt - 1), 10.0))

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """One API call, retried on transport errors, 429s and 5xx responses."""
        last_problem = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._http.request(
                    method, self._url(path), headers=self._headers(), json=payload
                )
            except httpx.TransportError as exc:
                last_problem = f"{exc.__class__.__name__}: {exc}"
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise RunpodError(
                    f"Could not reach Runpod at {self.base_url} ({last_problem}). "
                    "Check your network connection and RUNPOD_BASE_URL."
                ) from exc

            if response.status_code in (401, 403):
                raise RunpodError(
                    f"Runpod rejected the API key (HTTP {response.status_code}). "
                    f"Key in use: {self.masked_api_key}. "
                    "Re-check RUNPOD_API_KEY at https://www.runpod.io/console/user/settings."
                )

            if response.status_code == 404:
                raise RunpodError(
                    f"Endpoint '{self.endpoint_id}' was not found (HTTP 404). "
                    "Check RUNPOD_ENDPOINT_ID, and make sure the endpoint is deployed "
                    "and not paused."
                )

            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_problem = f"HTTP {response.status_code}: {_error_detail(response)}"
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise RunpodError(f"Runpod is unavailable ({last_problem}). Try again shortly.")

            if response.status_code >= 400:
                raise RunpodError(
                    f"Runpod returned HTTP {response.status_code}: {_error_detail(response)}"
                )

            try:
                body = response.json()
            except ValueError as exc:
                raise RunpodError("Runpod returned a response that was not JSON.") from exc

            if not isinstance(body, dict):
                raise RunpodError(f"Unexpected Runpod response: {str(body)[:200]}")
            return body

        raise RunpodError(last_problem or "Runpod request failed.")

    # ── API ──────────────────────────────────────────────────────────────────
    def health(self) -> dict:
        """Endpoint health (workers + queue depth). Raises if not deployed."""
        return self._request("GET", "health")

    def submit(self, payload: dict) -> str:
        """Queue a job and return its id."""
        body = self._request("POST", "run", payload)
        job_id = body.get("id")
        if not job_id:
            raise RunpodError(f"Runpod did not return a job id: {json.dumps(body)[:300]}")
        return str(job_id)

    def status(self, job_id: str) -> dict:
        return self._request("GET", f"status/{job_id}")

    def cancel(self, job_id: str) -> None:
        """Best-effort cancel; a job that already finished cannot be cancelled."""
        try:
            self._request("POST", f"cancel/{job_id}")
        except RunpodError:
            pass

    def wait(self, job_id: str, timeout_s: float | None = None) -> JobResult:
        """Poll until the job finishes, fails, or the timeout expires."""
        budget = self.timeout_s if timeout_s is None else timeout_s
        deadline = self._clock() + max(1.0, float(budget))

        while True:
            body = self.status(job_id)
            state = str(body.get("status") or "").upper()

            if state == TERMINAL_OK:
                return JobResult(
                    job_id=job_id,
                    status=state,
                    output=body.get("output"),
                    delay_time_ms=_numeric(body.get("delayTime")),
                    execution_time_ms=_numeric(body.get("executionTime")),
                    raw=body,
                )

            if state in TERMINAL_BAD:
                detail = body.get("error") or body.get("output") or "no error detail returned"
                raise RunpodJobError(
                    f"Runpod job {job_id} ended as {state}: {str(detail)[:500]}",
                    job_id=job_id,
                    status=state,
                    output=body.get("output"),
                )

            if self._clock() >= deadline:
                self.cancel(job_id)
                raise RunpodTimeout(
                    f"Runpod job {job_id} did not finish within {budget:.0f}s. "
                    "The job was cancelled; increase RUNPOD_JOB_TIMEOUT_S for long renders.",
                    job_id=job_id,
                )

            self._sleep(self.poll_interval_s)

    def run(self, payload: dict, timeout_s: float | None = None) -> JobResult:
        """Submit a job and block until it completes."""
        return self.wait(self.submit(payload), timeout_s=timeout_s)
