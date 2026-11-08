"""HTTP surface: health, models, media and the Haught Studio chat route."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from runpod_comfyui_mcp import server as srv
from runpod_comfyui_mcp.config import Settings

TEMPLATE = Path(__file__).resolve().parents[1] / "workflows" / "krea2_turbo_t2i.json"
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def client():
    app = srv.server.streamable_http_app()
    with TestClient(app) as test_client:
        yield test_client
    srv.configure(None, "reset by test fixture")


def test_health_reports_missing_configuration(client):
    srv.configure(None, "RUNPOD_API_KEY is not set.")

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["configured"] is False
    assert "RUNPOD_API_KEY" in body["error"]


def test_health_masks_the_api_key(client, tmp_path):
    srv.configure(
        Settings(
            api_key="rpa_supersecret_value",
            endpoint_id="ep123",
            output_dir=tmp_path,
            workflow_path=TEMPLATE,
            transport="streamable-http",
        )
    )

    body = client.get("/health").json()

    assert body["configured"] is True
    assert body["runpod_api_key"] == "rpa_...alue"
    assert "rpa_supersecret_value" not in str(body)


def test_models_lists_the_connector_model(client):
    body = client.get("/v1/models").json()
    assert body["data"][0]["id"] == "runpod-krea2-turbo"


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "runpod-comfyui-mcp" in response.text


def test_media_serves_files_from_the_output_folder(client, tmp_path):
    srv.configure(
        Settings(
            api_key="k",
            endpoint_id="ep",
            output_dir=tmp_path,
            workflow_path=TEMPLATE,
            transport="streamable-http",
        )
    )
    (tmp_path / "shot.png").write_bytes(PNG_BYTES)

    response = client.get("/media/shot.png")

    assert response.status_code == 200
    assert response.content == PNG_BYTES


def test_media_rejects_traversal_and_unknown_files(client, tmp_path):
    srv.configure(
        Settings(
            api_key="k",
            endpoint_id="ep",
            output_dir=tmp_path / "output",
            workflow_path=TEMPLATE,
            transport="streamable-http",
        )
    )
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")

    assert client.get("/media/..%2Fsecret.txt").status_code == 404
    assert client.get("/media/absent.png").status_code == 404


def test_chat_completions_requires_a_prompt(client):
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_prompt"


def test_chat_completions_returns_a_chat_envelope(client, tmp_path, monkeypatch):
    srv.configure(
        Settings(
            api_key="k",
            endpoint_id="ep",
            output_dir=tmp_path,
            workflow_path=TEMPLATE,
            transport="streamable-http",
            port=8600,
        )
    )
    image_path = tmp_path / "fox.png"
    image_path.write_bytes(PNG_BYTES)

    def fake_generate(**kwargs):
        assert kwargs["prompt"] == "a red fox"
        return {
            "ok": True,
            "images": [
                {
                    "filename": "fox.png",
                    "path": str(image_path),
                    "mime_type": "image/png",
                    "bytes": len(PNG_BYTES),
                    "url": "http://127.0.0.1:8600/media/fox.png",
                }
            ],
        }

    monkeypatch.setattr(srv, "generate", fake_generate)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "runpod-krea2-turbo", "messages": [{"role": "user", "content": "a red fox"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["images"][0] == base64.b64encode(PNG_BYTES).decode()
    assert "/media/fox.png" in message["content"]


def test_chat_completions_reports_generation_failure(client, monkeypatch):
    monkeypatch.setattr(srv, "generate", lambda **_kwargs: {"ok": False, "error": "runpod is down"})

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "a red fox"}]},
    )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "runpod is down"
