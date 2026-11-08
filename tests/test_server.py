from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from runpod_comfyui_mcp import server as srv
from runpod_comfyui_mcp.config import ConfigError, Settings
from runpod_comfyui_mcp.runpod import JobResult, RunpodJobError, RunpodTimeout

TEMPLATE = Path(__file__).resolve().parents[1] / "workflows" / "krea2_turbo_t2i.json"
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "api_key": "rpa_test_key",
        "endpoint_id": "ep123",
        "output_dir": tmp_path,
        "workflow_path": TEMPLATE,
    }
    values.update(overrides)
    return Settings(**values)


class FakeClient:
    def __init__(self, *, output=None, run_error=None, health=None):
        self.output = output
        self.run_error = run_error
        self.health_body = health or {"workers": {"idle": 1, "running": 0}, "jobs": {"inQueue": 0}}
        self.payload = None
        self.cancelled: list[str] = []

    def run(self, payload, timeout_s=None):
        self.payload = payload
        if self.run_error:
            raise self.run_error
        return JobResult(
            job_id="job-1",
            status="COMPLETED",
            output=self.output,
            delay_time_ms=500,
            execution_time_ms=2400,
        )

    def health(self):
        return self.health_body

    def cancel(self, job_id):
        self.cancelled.append(job_id)


@pytest.fixture(autouse=True)
def reset_runtime():
    yield
    srv.configure(None, "reset by test fixture")


def test_generate_saves_the_returned_image(tmp_path, monkeypatch):
    srv.configure(make_settings(tmp_path))
    client = FakeClient(output={"images": [{"filename": "fox.png", "type": "base64", "data": PNG_B64}]})
    monkeypatch.setattr(srv, "get_client", lambda: client)

    info = srv.generate(prompt="a red fox in snow", seed=7)

    assert info["ok"] is True
    assert (tmp_path / "fox.png").is_file()
    assert info["images"][0]["filename"] == "fox.png"
    assert info["images"][0]["url"] is None
    assert info["job"]["id"] == "job-1"
    assert info["job"]["execution_time_s"] == 2.4
    assert client.payload["input"]["workflow"]["120"]["inputs"]["value"] == 7


def test_generate_returns_http_urls_in_http_mode(tmp_path, monkeypatch):
    srv.configure(make_settings(tmp_path, transport="streamable-http", port=8600))
    client = FakeClient(output={"images": [{"filename": "fox.png", "type": "base64", "data": PNG_B64}]})
    monkeypatch.setattr(srv, "get_client", lambda: client)

    info = srv.generate(prompt="a red fox")

    assert info["images"][0]["url"] == "http://127.0.0.1:8600/media/fox.png"


def test_generate_without_configuration_is_a_clear_error():
    srv.configure(None, "RUNPOD_API_KEY is not set.")

    info = srv.generate(prompt="anything")

    assert info["ok"] is False
    assert "RUNPOD_API_KEY" in info["error"]
    assert "hint" in info


def test_generate_rejects_an_empty_prompt(tmp_path):
    srv.configure(make_settings(tmp_path))

    info = srv.generate(prompt="   ")

    assert info["ok"] is False
    assert "prompt" in info["error"].lower()


def test_generate_rejects_out_of_range_values(tmp_path):
    srv.configure(make_settings(tmp_path))

    assert srv.generate(prompt="x", steps=0)["ok"] is False
    assert srv.generate(prompt="x", cfg=99)["ok"] is False
    assert srv.generate(prompt="x", width=10)["ok"] is False


def test_generate_requires_latent_friendly_sizes(tmp_path):
    srv.configure(make_settings(tmp_path))

    info = srv.generate(prompt="x", width=1002)

    assert info["ok"] is False
    assert "multiple of 8" in info["error"]


def test_generate_reports_job_failure(tmp_path, monkeypatch):
    srv.configure(make_settings(tmp_path))
    client = FakeClient(run_error=RunpodJobError("boom", job_id="job-9", status="FAILED"))
    monkeypatch.setattr(srv, "get_client", lambda: client)

    info = srv.generate(prompt="x")

    assert info["ok"] is False
    assert info["job_id"] == "job-9"
    assert info["job_status"] == "FAILED"


def test_generate_reports_timeout(tmp_path, monkeypatch):
    srv.configure(make_settings(tmp_path))
    client = FakeClient(run_error=RunpodTimeout("too slow", job_id="job-8"))
    monkeypatch.setattr(srv, "get_client", lambda: client)

    info = srv.generate(prompt="x")

    assert info["ok"] is False
    assert info["job_id"] == "job-8"


def test_generate_reports_an_empty_job_result(tmp_path, monkeypatch):
    srv.configure(make_settings(tmp_path))
    client = FakeClient(output={"images": []})
    monkeypatch.setattr(srv, "get_client", lambda: client)

    info = srv.generate(prompt="x")

    assert info["ok"] is False
    assert "no image" in info["error"]


def test_check_setup_reports_config_problems(monkeypatch):
    def boom():
        raise ConfigError("RUNPOD_API_KEY is not set.")

    monkeypatch.setattr(srv, "load_settings", boom)

    report = srv.check_setup()

    assert report["ok"] is False
    assert "RUNPOD_API_KEY" in report["error"]


def test_check_setup_reports_workflow_and_endpoint(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setattr(srv, "load_settings", lambda: settings)
    monkeypatch.setattr(srv, "get_client", lambda: FakeClient())

    report = srv.check_setup()

    assert report["ok"] is True
    assert report["workflow"]["valid"] is True
    assert report["endpoint"]["reachable"] is True
    assert "rpa_test_key" not in str(report)


def test_cancel_job_delegates(tmp_path, monkeypatch):
    srv.configure(make_settings(tmp_path))
    client = FakeClient()
    monkeypatch.setattr(srv, "get_client", lambda: client)

    assert srv.cancel_job("job-1")["ok"] is True
    assert client.cancelled == ["job-1"]


def test_prompt_extraction_from_chat_messages():
    prompt, extra = srv._prompt_from_messages([{"role": "user", "content": "a cat"}])
    assert prompt == "a cat"
    assert extra == {}

    prompt, extra = srv._prompt_from_messages(
        [{"role": "user", "content": '{"prompt": "a dog", "steps": 4}'}]
    )
    assert prompt == "a dog"
    assert extra["steps"] == 4

    prompt, _ = srv._prompt_from_messages(
        [{"role": "user", "content": [{"type": "text", "text": "a bird"}]}]
    )
    assert prompt == "a bird"


def test_tools_are_registered():
    tools = asyncio.run(srv.server.list_tools())
    names = {tool.name for tool in tools}
    assert {"generate_image", "check_setup", "cancel_job"} <= names
