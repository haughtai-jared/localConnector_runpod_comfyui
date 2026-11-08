from __future__ import annotations

import json

import httpx
import pytest

from runpod_comfyui_mcp.runpod import (
    RunpodClient,
    RunpodError,
    RunpodJobError,
    RunpodTimeout,
)


class FakeTime:
    """Deterministic clock so timeout tests do not actually wait."""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_client(handler, **kwargs) -> RunpodClient:
    fake = FakeTime()
    return RunpodClient(
        api_key=kwargs.pop("api_key", "rpa_test_key_1234567890"),
        endpoint_id=kwargs.pop("endpoint_id", "ep123"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=fake.sleep,
        clock=fake.clock,
        **kwargs,
    )


def test_submit_and_wait_returns_output():
    calls: list[str] = []
    polls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path.endswith("/run"):
            assert request.headers["Authorization"] == "Bearer rpa_test_key_1234567890"
            return httpx.Response(200, json={"id": "job-1", "status": "IN_QUEUE"})
        polls["count"] += 1
        if polls["count"] == 1:
            return httpx.Response(200, json={"id": "job-1", "status": "IN_PROGRESS"})
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "status": "COMPLETED",
                "output": {"images": [{"filename": "a.png", "type": "base64", "data": "AAAA"}]},
                "delayTime": 1500,
                "executionTime": 4300,
            },
        )

    client = make_client(handler)
    result = client.run({"input": {"workflow": {}}})

    assert result.job_id == "job-1"
    assert result.status == "COMPLETED"
    assert result.delay_time_ms == 1500
    assert result.execution_time_ms == 4300
    assert result.output["images"][0]["filename"] == "a.png"
    assert calls == ["POST /v2/ep123/run", "GET /v2/ep123/status/job-1", "GET /v2/ep123/status/job-1"]


def test_missing_job_id_is_an_error():
    client = make_client(lambda request: httpx.Response(200, json={"status": "IN_QUEUE"}))
    with pytest.raises(RunpodError, match="job id"):
        client.submit({"input": {}})


def test_failed_job_reports_the_worker_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/run"):
            return httpx.Response(200, json={"id": "job-2", "status": "IN_QUEUE"})
        return httpx.Response(
            200,
            json={"id": "job-2", "status": "FAILED", "error": "CUDA out of memory"},
        )

    client = make_client(handler)
    with pytest.raises(RunpodJobError) as excinfo:
        client.run({"input": {}})

    assert "CUDA out of memory" in str(excinfo.value)
    assert excinfo.value.job_id == "job-2"
    assert excinfo.value.status == "FAILED"


def test_timeout_cancels_the_job():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/run"):
            return httpx.Response(200, json={"id": "job-3"})
        if request.url.path.endswith("/cancel/job-3"):
            return httpx.Response(200, json={"id": "job-3", "status": "CANCELLED"})
        return httpx.Response(200, json={"id": "job-3", "status": "IN_PROGRESS"})

    client = make_client(handler, timeout_s=10, poll_interval_s=2)
    with pytest.raises(RunpodTimeout) as excinfo:
        client.run({"input": {}})

    assert excinfo.value.job_id == "job-3"
    assert any(path.endswith("/cancel/job-3") for path in paths)


def test_auth_failure_mentions_masked_key_not_the_secret():
    client = make_client(lambda request: httpx.Response(401, json={"error": "unauthorized"}))
    with pytest.raises(RunpodError) as excinfo:
        client.submit({"input": {}})

    message = str(excinfo.value)
    assert "rejected the API key" in message
    assert "rpa_test_key_1234567890" not in message
    assert "rpa_...7890" in message


def test_unknown_endpoint_reports_404():
    client = make_client(lambda request: httpx.Response(404, json={"error": "not found"}))
    with pytest.raises(RunpodError, match="not found"):
        client.submit({"input": {}})


def test_server_errors_are_retried_then_succeed():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, json={"error": "warming up"})
        return httpx.Response(200, json={"id": "job-4"})

    client = make_client(handler, max_attempts=3)
    assert client.submit({"input": {}}) == "job-4"
    assert attempts["count"] == 2


def test_transport_errors_are_retried_then_raise():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("no route to host", request=request)

    client = make_client(handler, max_attempts=2)
    with pytest.raises(RunpodError, match="Could not reach Runpod"):
        client.submit({"input": {}})
    assert attempts["count"] == 2


def test_client_errors_are_not_retried():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(400, json={"error": "workflow is invalid"})

    client = make_client(handler)
    with pytest.raises(RunpodError, match="workflow is invalid"):
        client.submit({"input": {}})
    assert attempts["count"] == 1


def test_health_returns_worker_counts():
    client = make_client(
        lambda request: httpx.Response(
            200, json={"workers": {"idle": 1, "running": 0}, "jobs": {"inQueue": 0}}
        )
    )
    assert client.health()["workers"]["idle"] == 1


def test_empty_credentials_are_rejected():
    with pytest.raises(RunpodError):
        RunpodClient(api_key="", endpoint_id="ep")
    with pytest.raises(RunpodError):
        RunpodClient(api_key="key", endpoint_id="")


def test_error_detail_falls_back_to_raw_text():
    client = make_client(lambda request: httpx.Response(400, text="plain text failure"))
    with pytest.raises(RunpodError, match="plain text failure"):
        client.submit({"input": {}})


def test_payload_is_sent_as_json():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"id": "job-5"})

    make_client(handler).submit({"input": {"workflow": {"1": {}}}})
    assert captured == {"input": {"workflow": {"1": {}}}}
