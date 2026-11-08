from __future__ import annotations

import base64

import pytest

from runpod_comfyui_mcp.outputs import (
    OutputError,
    decode_payload,
    iter_output_items,
    sanitize_filename,
    save_outputs,
    sniff,
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_sniff_detects_png_and_rejects_unknown():
    assert sniff(PNG_BYTES) == (".png", "image/png")
    assert sniff(b"not an image") is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("frame.png", "frame.png"),
        ("../../etc/passwd", "passwd"),
        ("C:\\temp\\shot.JPG", "shot.JPG"),
        ("", "fallback.png"),
        ("...", "fallback.png"),
        ("weird name?.png", "weird_name_.png"),
    ],
)
def test_sanitize_filename(name, expected):
    assert sanitize_filename(name, fallback="fallback.png") == expected


@pytest.mark.parametrize(
    "output",
    [
        {"images": [{"filename": "a.png", "type": "base64", "data": "AAAA"}]},
        [{"filename": "a.png", "type": "base64", "data": "AAAA"}],
        {"images": ["AAAA"]},
        {"output": {"images": [{"data": "AAAA"}]}},
    ],
)
def test_iter_output_items_handles_every_shape(output):
    items = list(iter_output_items(output))
    assert len(items) == 1
    assert items[0]["data"] == "AAAA"
    assert items[0]["kind"] == "base64"


def test_iter_output_items_detects_urls():
    items = list(iter_output_items({"images": [{"filename": "a.png", "type": "s3_url", "data": "https://x/y.png"}]}))
    assert items[0]["kind"] == "url"


def test_decode_payload_accepts_data_urls_and_whitespace():
    encoded = base64.b64encode(PNG_BYTES).decode()
    assert decode_payload(f"data:image/png;base64,{encoded}") == PNG_BYTES
    assert decode_payload(encoded[:20] + "\n" + encoded[20:]) == PNG_BYTES


def test_decode_payload_rejects_empty():
    with pytest.raises(OutputError):
        decode_payload("")


def test_save_outputs_writes_sanitized_files(tmp_path):
    encoded = base64.b64encode(PNG_BYTES).decode()

    saved, problems = save_outputs(
        {"images": [{"filename": "../../evil/../cat.png", "type": "base64", "data": encoded}]},
        tmp_path,
    )

    assert problems == []
    assert len(saved) == 1
    assert saved[0].filename == "cat.png"
    assert saved[0].path.parent == tmp_path
    assert saved[0].mime_type == "image/png"
    assert saved[0].path.read_bytes() == PNG_BYTES


def test_save_outputs_avoids_collisions(tmp_path):
    encoded = base64.b64encode(PNG_BYTES).decode()
    output = {"images": [{"filename": "same.png", "type": "base64", "data": encoded}] * 2}

    saved, _ = save_outputs(output, tmp_path)

    assert [item.filename for item in saved] == ["same.png", "same-1.png"]


def test_save_outputs_downloads_urls(tmp_path):
    requested: list[str] = []

    def downloader(url: str) -> bytes:
        requested.append(url)
        return PNG_BYTES

    saved, problems = save_outputs(
        {"images": [{"filename": "x", "type": "s3_url", "data": "https://cdn.example/x"}]},
        tmp_path,
        downloader=downloader,
    )

    assert requested == ["https://cdn.example/x"]
    assert problems == []
    assert saved[0].filename == "x.png"
    assert saved[0].source == "url"


def test_save_outputs_keeps_going_after_a_bad_entry(tmp_path):
    encoded = base64.b64encode(PNG_BYTES).decode()

    saved, problems = save_outputs(
        {
            "images": [
                {"filename": "bad.png", "type": "base64", "data": "@@not base64@@"},
                {"filename": "good.png", "type": "base64", "data": encoded},
            ]
        },
        tmp_path,
    )

    assert len(saved) == 1
    assert saved[0].filename == "good.png"
    assert len(problems) == 1


def test_save_outputs_returns_nothing_for_empty_output(tmp_path):
    saved, problems = save_outputs({"images": []}, tmp_path)
    assert saved == []
    assert problems == []


def test_download_rejects_non_http_urls():
    from runpod_comfyui_mcp.outputs import download

    with pytest.raises(OutputError, match="non-HTTP"):
        download("file:///etc/passwd")
