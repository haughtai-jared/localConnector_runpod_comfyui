"""The MCP server: tools plus a small HTTP surface.

Two transports are supported, chosen at startup:

* ``streamable-http`` — how Haught Studio reaches the connector: MCP at ``/mcp``,
  plus the chat route (``/v1/chat/completions``), ``/media`` and a small usage page.
* ``stdio`` — for a harness that launches the connector as a child process.
"""

from __future__ import annotations

import base64
import functools
import html
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response

from . import __version__
from .config import ConfigError, Settings, load_settings
from .outputs import MIME_TYPES, SavedFile, save_outputs
from .runpod import RunpodClient, RunpodError, RunpodJobError, RunpodTimeout
from .workflow import TextToImageParams, WorkflowError, build_text_to_image, describe_template

logger = logging.getLogger("runpod_comfyui_mcp")

MODEL_ID = "runpod-krea2-turbo"

INSTRUCTIONS = (
    "This connector renders images with a ComfyUI workflow executed on a Runpod "
    "serverless GPU endpoint. The workflow JSON is prepared locally, the reference "
    "files (if any) are embedded in the job payload, and the job is submitted to "
    "Runpod with /run, then polled until it completes.\n\n"
    "Call `generate_image` with a detailed `prompt`. Optional fields: "
    "`negative_prompt`, `width`, `height`, `steps`, `cfg`, `sampler`, `scheduler`, "
    "`seed`, `lora_strength`, `filename_prefix`, and `timeout_s`.\n"
    "Generated files are saved to the connector's output folder; the reply includes "
    "their local paths (and HTTP URLs when the server runs in streamable-http mode).\n"
    "Call `check_setup` first if you are unsure whether Runpod credentials, the "
    "endpoint, or the workflow template are configured correctly."
)

server = MCPServer(
    name="runpod-comfyui-mcp",
    title="Runpod ComfyUI MCP",
    version=__version__,
    instructions=INSTRUCTIONS,
)

# ── runtime state ────────────────────────────────────────────────────────────
_state: dict[str, Any] = {"settings": None, "client": None, "error": None}
_state_lock = threading.Lock()


def configure(settings: Settings | None, error: str | None = None) -> None:
    """Install validated settings (or the reason they are missing)."""
    with _state_lock:
        _state["settings"] = settings
        _state["error"] = error
        if _state["client"] is not None:
            try:
                _state["client"].close()
            except Exception:  # pragma: no cover - closing a socket must never raise here
                pass
        _state["client"] = None


def get_settings() -> Settings:
    settings = _state["settings"]
    if settings is None:
        raise ConfigError(_state["error"] or "Runpod is not configured.")
    return settings


def get_client() -> RunpodClient:
    with _state_lock:
        client = _state["client"]
        if client is not None:
            return client
        settings = get_settings()
        _state["client"] = RunpodClient(
            api_key=settings.api_key,
            endpoint_id=settings.endpoint_id,
            base_url=settings.base_url,
            timeout_s=settings.job_timeout_s,
            poll_interval_s=settings.poll_interval_s,
        )
        return _state["client"]


def _config_error_result(exc: Exception) -> dict:
    return {
        "ok": False,
        "error": str(exc),
        "hint": (
            "Copy .env.example to .env and set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID. "
            "Run `runpod-comfyui-mcp --check` to validate the whole configuration."
        ),
    }


# ── validation helpers ───────────────────────────────────────────────────────
def _clamp_int(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"`{name}` must be an integer.")
    if not low <= value <= high:
        raise ValueError(f"`{name}` must be between {low} and {high}; got {value}.")
    return value


def _clamp_float(value: float | None, name: str, low: float, high: float) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be a number.") from exc
    if not low <= number <= high:
        raise ValueError(f"`{name}` must be between {low} and {high}; got {number}.")
    return number


def _image_entries(saved: list[SavedFile], settings: Settings | None) -> list[dict]:
    base = settings.media_base_url() if settings else ""
    entries = []
    for item in saved:
        entry = item.as_dict()
        entry["url"] = f"{base}/media/{item.filename}" if base else None
        entries.append(entry)
    return entries


# ── core operation (shared by the tool and the chat route) ──────────────────
def generate(
    *,
    prompt: str,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    seed: int | None = None,
    lora_strength: float | None = None,
    filename_prefix: str | None = None,
    timeout_s: float | None = None,
) -> dict:
    """Build the workflow, run it on Runpod, and save the results."""
    try:
        settings = get_settings()
    except ConfigError as exc:
        return _config_error_result(exc)

    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "`prompt` is required — describe the image you want."}

    try:
        params = TextToImageParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=_clamp_int(width, "width", 64, 4096),
            height=_clamp_int(height, "height", 64, 4096),
            steps=_clamp_int(steps, "steps", 1, 200),
            cfg=_clamp_float(cfg, "cfg", 0.0, 50.0),
            sampler=sampler,
            scheduler=scheduler,
            seed=seed,
            lora_strength=_clamp_float(lora_strength, "lora_strength", -5.0, 5.0),
            filename_prefix=filename_prefix,
        )
        for name, value in (("width", params.width), ("height", params.height)):
            if value is not None and value % 8:
                return {"ok": False, "error": f"`{name}` must be a multiple of 8; got {value}."}
        budget = _clamp_float(timeout_s, "timeout_s", 10.0, 7200.0) or settings.job_timeout_s
        prepared = build_text_to_image(settings.workflow_path, params)
    except (ValueError, WorkflowError) as exc:
        return {"ok": False, "error": str(exc)}

    payload = prepared.payload(output_type="base64")
    started = time.time()
    logger.info(
        "submitting job | endpoint=%s | workflow=%s | prompt=%s",
        settings.endpoint_id,
        Path(settings.workflow_path).name,
        prompt[:120],
    )

    try:
        result = get_client().run(payload, timeout_s=budget)
    except RunpodTimeout as exc:
        return {"ok": False, "error": str(exc), "job_id": exc.job_id, "workflow": prepared.report}
    except RunpodJobError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "job_id": exc.job_id,
            "job_status": exc.status,
            "workflow": prepared.report,
        }
    except RunpodError as exc:
        return {"ok": False, "error": str(exc), "workflow": prepared.report}

    elapsed = time.time() - started
    saved, problems = save_outputs(result.output, settings.output_dir, stem="krea2")
    if not saved:
        detail = "; ".join(problems) or "the worker returned no files"
        return {
            "ok": False,
            "error": f"Runpod job {result.job_id} completed but no image could be saved ({detail}).",
            "job_id": result.job_id,
            "job_status": result.status,
            "raw_output_preview": json.dumps(result.output)[:600],
            "workflow": prepared.report,
        }

    logger.info(
        "job %s finished in %.1fs | %d file(s) -> %s",
        result.job_id,
        elapsed,
        len(saved),
        settings.output_dir,
    )

    response: dict[str, Any] = {
        "ok": True,
        "images": _image_entries(saved, settings),
        "job": {
            "id": result.job_id,
            "status": result.status,
            "wall_clock_s": round(elapsed, 1),
            "execution_time_s": round((result.execution_time_ms or 0) / 1000, 1),
            "delay_time_s": round((result.delay_time_ms or 0) / 1000, 1),
        },
        "workflow": prepared.report,
    }
    if problems:
        response["warnings"] = problems
    return response


# ── MCP tools ────────────────────────────────────────────────────────────────
@server.tool()
def generate_image(
    prompt: str,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    seed: int | None = None,
    lora_strength: float | None = None,
    filename_prefix: str | None = None,
    timeout_s: float | None = None,
) -> dict:
    """Render an image from a text prompt on the Runpod ComfyUI endpoint.

    Only `prompt` is required. Anything left unset keeps the workflow template's
    value, so the defaults live in `workflows/krea2_turbo_t2i.json` rather than in
    this signature. The call blocks until the job finishes and returns the local
    path of every generated file.
    """
    return generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        scheduler=scheduler,
        seed=seed,
        lora_strength=lora_strength,
        filename_prefix=filename_prefix,
        timeout_s=timeout_s,
    )


@server.tool()
def check_setup(check_endpoint: bool = True) -> dict:
    """Validate credentials, the workflow template and the Runpod endpoint.

    Costs no GPU time. Use it whenever `generate_image` reports a configuration
    problem, or after editing `.env` / the workflow file.
    """
    report: dict[str, Any] = {
        "ok": False,
        "server": {"name": "runpod-comfyui-mcp", "version": __version__},
        "config": {},
        "workflow": {},
        "endpoint": {},
    }

    try:
        settings = load_settings()
    except ConfigError as exc:
        report["error"] = str(exc)
        report["hint"] = "Copy .env.example to .env, then re-run check_setup."
        return report

    report["config"] = settings.summary()
    report["workflow"] = describe_template(settings.workflow_path)

    if check_endpoint:
        try:
            health = get_client().health()
            report["endpoint"] = {
                "reachable": True,
                "workers": health.get("workers", health),
                "jobs": health.get("jobs", {}),
            }
        except RunpodError as exc:
            report["endpoint"] = {"reachable": False, "error": str(exc)}

    report["ok"] = bool(
        report["workflow"].get("valid")
        and (not check_endpoint or report["endpoint"].get("reachable"))
    )
    return report


@server.tool()
def cancel_job(job_id: str) -> dict:
    """Cancel a queued or running Runpod job by id."""
    try:
        get_client().cancel(job_id)
    except ConfigError as exc:
        return _config_error_result(exc)
    except RunpodError as exc:
        return {"ok": False, "error": str(exc), "job_id": job_id}
    return {"ok": True, "job_id": job_id, "status": "cancel requested"}


# ── HTTP surface (streamable-http transport only) ────────────────────────────
def _settings_or_none() -> Settings | None:
    settings = _state["settings"]
    return settings if isinstance(settings, Settings) else None


@server.custom_route("/health", methods=["GET"])
async def health_route(_request: Request) -> Response:
    configured = _state["settings"] is not None
    body = {
        "status": "ok",
        "server": "runpod-comfyui-mcp",
        "version": __version__,
        "configured": configured,
    }
    settings = _settings_or_none()
    if settings is not None:
        body.update(
            {
                "runpod_endpoint_id": settings.endpoint_id,
                "runpod_api_key": settings.masked_api_key,
                "workflow": str(settings.workflow_path),
                "workflow_exists": settings.workflow_path.is_file(),
                "output_dir": str(settings.output_dir),
                "transport": settings.transport,
            }
        )
    else:
        body["error"] = _state["error"]
    return JSONResponse(body)


@server.custom_route("/v1/models", methods=["GET"])
async def models_route(_request: Request) -> Response:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "runpod-comfyui-mcp",
                }
            ],
        }
    )


def _prompt_from_messages(messages: list[Any]) -> tuple[str, dict]:
    """Pull a prompt (and optional JSON arguments) out of a chat payload."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        else:
            text = ""
        if not text:
            continue
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except ValueError:
                return text, {}
            if isinstance(parsed, dict):
                return str(parsed.get("prompt") or ""), parsed
        return text, {}
    return "", {}


@server.custom_route("/v1/chat/completions", methods=["POST"])
async def chat_completions_route(request: Request) -> Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    prompt, embedded = _prompt_from_messages(payload.get("messages") or [])
    prompt = str(payload.get("prompt") or embedded.get("prompt") or prompt).strip()
    if not prompt:
        return JSONResponse(
            {
                "error": {
                    "message": "No prompt found. Send it as the last user message or a top-level `prompt`.",
                    "type": "invalid_request_error",
                    "code": "missing_prompt",
                }
            },
            status_code=400,
        )

    allowed = ("negative_prompt", "width", "height", "steps", "cfg", "sampler", "scheduler", "seed", "lora_strength", "filename_prefix", "timeout_s")
    kwargs = {key: payload[key] for key in allowed if payload.get(key) is not None}
    for key in allowed:
        if embedded.get(key) is not None:
            kwargs.setdefault(key, embedded[key])

    info = await anyio.to_thread.run_sync(functools.partial(generate, prompt=prompt, **kwargs))
    if not info.get("ok"):
        return JSONResponse(
            {"error": {"message": info.get("error", "generation failed"), "type": "server_error"}},
            status_code=502,
        )

    first = info["images"][0]
    lines = [f"![generated image]({first['url']})" if first.get("url") else f"Saved to `{first['path']}`"]
    lines.append("")
    for image in info["images"]:
        target = image.get("url") or image["path"]
        lines.append(f"- [{image['filename']}]({target})")
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(lines)}

    try:
        data = Path(first["path"]).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        message["images"] = [encoded]
        message["image_url"] = {"url": f"data:{first['mime_type']};base64,{encoded}"}
    except OSError:  # pragma: no cover - file vanished between save and reply
        pass

    usage = max(1, len(prompt) // 4)
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage, "completion_tokens": 0, "total_tokens": usage},
        }
    )


@server.custom_route("/media/{filename}", methods=["GET"])
async def media_route(request: Request) -> Response:
    filename = Path(str(request.path_params.get("filename", ""))).name
    settings = _settings_or_none()
    if settings is None or not filename:
        return JSONResponse({"error": "not found"}, status_code=404)

    path = (settings.output_dir / filename).resolve()
    if path.parent != settings.output_dir.resolve() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)

    return FileResponse(path, media_type=MIME_TYPES.get(path.suffix.lower(), "application/octet-stream"))


@server.custom_route("/", methods=["GET"])
async def index_route(_request: Request) -> Response:
    settings = _settings_or_none()
    configured = settings is not None

    recent: list[Path] = []
    if settings is not None and settings.output_dir.is_dir():
        recent = sorted(
            (p for p in settings.output_dir.iterdir() if p.suffix.lower() in MIME_TYPES),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:6]

    gallery = "\n".join(
        f'<figure><img src="/media/{html.escape(p.name)}" alt="{html.escape(p.name)}">'
        f'<figcaption><a href="/media/{html.escape(p.name)}">{html.escape(p.name)}</a></figcaption></figure>'
        for p in recent
    ) or "<p><em>No images rendered yet.</em></p>"

    config_block = (
        html.escape(json.dumps(settings.summary(), indent=2))
        if settings
        else html.escape(_state["error"] or "Not configured.")
    )

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>runpod-comfyui-mcp</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; background:#0f1115; color:#d8dee9;
         max-width: 62rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  h1 {{ font-weight: 600; margin-bottom: .25rem; }}
  h2 {{ margin-top: 2rem; font-size: 1.05rem; text-transform: uppercase; letter-spacing: .08em; color:#8aa0b8; }}
  code, pre {{ background:#181c23; border-radius: 6px; }}
  code {{ padding: .1rem .35rem; }}
  pre {{ padding: .9rem 1rem; overflow-x: auto; font-size: .85rem; }}
  .muted {{ color:#8aa0b8; }}
  .badge {{ display:inline-block; padding:.1rem .5rem; border-radius:999px; font-size:.75rem; }}
  .ok {{ background:#12351f; color:#7ee2a8; }}
  .bad {{ background:#3a1618; color:#ff9b9b; }}
  figure {{ margin:0; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:1rem; }}
  img {{ width:100%; border-radius:8px; border:1px solid #232a35; display:block; }}
  figcaption {{ font-size:.8rem; margin-top:.35rem; word-break:break-all; }}
  a {{ color:#7cc4ff; }}
</style></head>
<body>
<h1>runpod-comfyui-mcp <span class="muted">v{__version__}</span></h1>
<p>
  <span class="badge {'ok' if configured else 'bad'}">{'configured' if configured else 'not configured'}</span>
  <span class="muted">Krea2 Turbo text-to-image on a Runpod serverless endpoint</span>
</p>

<h2>MCP</h2>
<pre>POST http://127.0.0.1:{settings.port if settings else 8600}/mcp
{{"jsonrpc":"2.0","id":1,"method":"tools/list"}}</pre>

<h2>Connect from Haught Studio</h2>
<pre>POST http://127.0.0.1:{settings.port if settings else 8600}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{{"model":"{MODEL_ID}","messages":[{{"role":"user","content":"a red fox in fresh snow, golden hour"}}]}}'</pre>

<h2>Configuration</h2>
<pre>{config_block}</pre>

<h2>Recent images</h2>
<div class="grid">{gallery}</div>
</body></html>"""
    return HTMLResponse(body)


def format_config_error(exc: Exception) -> str:
    return (
        f"{exc}\n\n"
        "The server is still running, but generation calls will fail until this is fixed.\n"
        "Validate with:  runpod-comfyui-mcp --check"
    )
