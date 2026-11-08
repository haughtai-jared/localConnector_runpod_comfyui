"""Extract, decode and save the files a Runpod ComfyUI worker returns.

The ComfyUI worker can answer in more than one shape depending on its version and
``output_type``:

``{"images": [{"filename": "x.png", "type": "base64", "data": "..."}]}``
``{"images": [{"filename": "x.png", "type": "s3_url", "data": "https://..."}]}``
``{"images": ["<base64>"]}``                              (older worker builds)

All of them are normalised here, and every file lands in the local output folder
with a sanitised, collision-free name.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (b"\xff\xd8\xff", ".jpg", "image/jpeg"),
    (b"GIF87a", ".gif", "image/gif"),
    (b"GIF89a", ".gif", "image/gif"),
    (b"BM", ".bmp", "image/bmp"),
)

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


class OutputError(RuntimeError):
    """A worker output entry could not be decoded or downloaded."""


@dataclass(frozen=True)
class SavedFile:
    path: Path
    filename: str
    mime_type: str
    bytes: int
    source: str = "base64"

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "path": str(self.path),
            "mime_type": self.mime_type,
            "bytes": self.bytes,
        }


def sniff(data: bytes) -> tuple[str, str] | None:
    """Return ``(extension, mime_type)`` from magic bytes."""
    for magic, extension, mime in _SIGNATURES:
        if data.startswith(magic):
            return extension, mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if data[4:8] == b"ftyp":
        return ".mp4", "video/mp4"
    return None


def sanitize_filename(name: str | None, *, fallback: str) -> str:
    """Reduce a worker-supplied name to a safe local basename."""
    candidate = Path(str(name or "")).name.strip()
    candidate = NAME_PATTERN.sub("_", candidate).strip("._")
    if not candidate:
        return fallback
    return candidate[:120]


def _kind(value: str) -> str:
    lowered = value.lstrip().lower()
    if lowered.startswith(("http://", "https://")):
        return "url"
    if lowered.startswith("data:"):
        return "data_url"
    return "base64"


def _visible(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def iter_output_items(output: Any) -> Iterator[dict]:
    """Yield normalised ``{filename, kind, data}`` entries from a worker output."""
    if output is None:
        return

    if isinstance(output, str):
        yield {"filename": None, "kind": _kind(output), "data": output.strip()}
        return

    if isinstance(output, (list, tuple)):
        for item in output:
            yield from iter_output_items(item)
        return

    if not isinstance(output, dict):
        return

    for key in ("images", "videos", "gifs", "files", "output"):
        nested = output.get(key)
        if isinstance(nested, (list, tuple, str, dict)):
            yield from iter_output_items(nested)
            return

    data = None
    for key in ("data", "image", "base64", "url", "s3_url"):
        data = _visible(output.get(key))
        if data:
            break
    if data is None:
        return

    declared = str(output.get("type") or "").lower()
    if declared in ("s3_url", "url", "http"):
        kind = "url"
    elif declared in ("base64", "b64", "data_url"):
        kind = "data_url" if data.lstrip().lower().startswith("data:") else "base64"
    else:
        kind = _kind(data)

    yield {
        "filename": _visible(output.get("filename")) or _visible(output.get("name")),
        "kind": kind,
        "data": data,
    }


def decode_payload(data: str) -> bytes:
    """Decode a base64 blob or ``data:`` URL, tolerating whitespace and URL-safe text."""
    payload = data.strip()
    if payload.lower().startswith("data:"):
        _, _, payload = payload.partition(",")
    payload = re.sub(r"\s+", "", payload)
    payload = payload.replace("-", "+").replace("_", "/")
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise OutputError(f"Could not decode the base64 payload ({exc}).") from exc
    if not decoded:
        raise OutputError("The worker returned an empty payload.")
    return decoded


def download(url: str, *, timeout_s: float = 120.0) -> bytes:
    """Fetch a signed result URL. Only http(s) is allowed."""
    lowered = url.lower()
    if not lowered.startswith(("http://", "https://")):
        raise OutputError(f"Refusing to download a non-HTTP result URL: {url[:80]}")

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=timeout_s) as response:
            if response.status_code >= 400:
                raise OutputError(f"Result download failed with HTTP {response.status_code}.")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise OutputError("Result file exceeds the download size limit.")
                chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise OutputError(f"Could not download the result file ({exc.__class__.__name__}).") from exc

    data = b"".join(chunks)
    if not data:
        raise OutputError("The result download was empty.")
    return data


def _unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise OutputError(f"Could not find a free filename for {filename}.")


def save_outputs(
    output: Any,
    output_dir: str | Path,
    *,
    stem: str = "image",
    downloader: Callable[[str], bytes] = download,
) -> tuple[list[SavedFile], list[str]]:
    """Write every image in ``output`` to ``output_dir``.

    Returns ``(saved, problems)``. A single bad entry does not discard the rest.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    saved: list[SavedFile] = []
    problems: list[str] = []

    for index, item in enumerate(iter_output_items(output), 1):
        try:
            if item["kind"] == "url":
                data = downloader(item["data"])
                source = "url"
            else:
                data = decode_payload(item["data"])
                source = "base64"
        except OutputError as exc:
            problems.append(f"output[{index}]: {exc}")
            continue

        name = sanitize_filename(item.get("filename"), fallback=f"{stem}_{index}.png")
        detected = sniff(data)
        if detected:
            extension, mime = detected
            if Path(name).suffix.lower() not in MIME_TYPES or Path(name).suffix.lower() != extension:
                name = f"{Path(name).stem or stem}{extension}"
        else:
            mime = MIME_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")

        path = _unique_path(directory, name)
        path.write_bytes(data)
        saved.append(
            SavedFile(
                path=path,
                filename=path.name,
                mime_type=mime,
                bytes=len(data),
                source=source,
            )
        )

    return saved, problems
