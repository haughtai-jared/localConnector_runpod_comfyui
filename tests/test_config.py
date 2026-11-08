from __future__ import annotations

import pytest

from runpod_comfyui_mcp.config import (
    ConfigError,
    DEFAULT_BASE_URL,
    Settings,
    load_settings,
    mask_secret,
)


def _base_env(**overrides: str) -> dict[str, str]:
    env = {
        "RUNPOD_API_KEY": "rpa_test_key_1234567890",
        "RUNPOD_ENDPOINT_ID": "abc123def456",
    }
    env.update(overrides)
    return env


def test_loads_required_values(tmp_path):
    workflow = tmp_path / "wf.json"
    workflow.write_text('{"1": {"class_type": "KSampler", "inputs": {}}}', encoding="utf-8")

    settings = load_settings(environ=_base_env(WORKFLOW_PATH=str(workflow)))

    assert settings.api_key == "rpa_test_key_1234567890"
    assert settings.endpoint_id == "abc123def456"
    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.endpoint_url == "https://api.runpod.ai/v2/abc123def456"
    assert settings.workflow_path == workflow.resolve()
    assert settings.transport == "stdio"
    assert settings.media_base_url() == ""


def test_missing_secrets_are_reported_together():
    with pytest.raises(ConfigError) as excinfo:
        load_settings(environ={})

    message = str(excinfo.value)
    assert "RUNPOD_API_KEY" in message
    assert "RUNPOD_ENDPOINT_ID" in message
    assert ".env.example" in message


def test_endpoint_id_must_not_be_a_url():
    with pytest.raises(ConfigError) as excinfo:
        load_settings(environ=_base_env(RUNPOD_ENDPOINT_ID="https://api.runpod.ai/v2/abc123"))
    assert "bare endpoint id" in str(excinfo.value)


def test_invalid_numbers_are_reported(tmp_path):
    workflow = tmp_path / "wf.json"
    workflow.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_settings(
            environ=_base_env(
                WORKFLOW_PATH=str(workflow),
                RUNPOD_POLL_INTERVAL_S="soon",
                MCP_PORT="eighty",
            )
        )
    message = str(excinfo.value)
    assert "RUNPOD_POLL_INTERVAL_S" in message
    assert "MCP_PORT" in message


def test_missing_workflow_is_reported(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_settings(environ=_base_env(WORKFLOW_PATH=str(tmp_path / "nope.json")))
    assert "Workflow template not found" in str(excinfo.value)


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RUNPOD_API_KEY=from_dotenv\nRUNPOD_ENDPOINT_ID=from_dotenv_endpoint\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "from_process")

    settings = load_settings(env_file=env_file)

    assert settings.api_key == "from_process"
    assert settings.endpoint_id == "from_dotenv_endpoint"


def test_missing_env_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(env_file=tmp_path / "absent.env")


def test_http_mode_builds_media_base_url():
    settings = Settings(
        api_key="k",
        endpoint_id="e",
        transport="streamable-http",
        host="0.0.0.0",
        port=9000,
    )
    assert settings.media_base_url() == "http://127.0.0.1:9000"

    public = Settings(
        api_key="k",
        endpoint_id="e",
        transport="streamable-http",
        public_base_url="https://example.test/",
    )
    assert public.media_base_url() == "https://example.test"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "<unset>"),
        ("", "<unset>"),
        ("short", "*****"),
        ("rpa_1234567890abcdef", "rpa_...cdef"),
    ],
)
def test_mask_secret(value, expected):
    assert mask_secret(value) == expected


def test_summary_never_contains_the_raw_key():
    settings = Settings(api_key="rpa_supersecret_value", endpoint_id="endpoint1")
    summary = settings.summary()
    assert "rpa_supersecret_value" not in str(summary)
    assert summary["runpod_api_key"] == "rpa_...alue"
