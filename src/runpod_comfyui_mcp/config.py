"""Configuration for the local connector.

Everything the server needs is read from the environment, with ``.env`` used only as
a fallback. Real environment variables always win, which is what you want when
Haught Studio (or a container) injects the secrets itself.

Nothing in this module ever logs or serialises the API key in full.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.runpod.ai/v2"
DEFAULT_JOB_TIMEOUT_S = 900.0
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_TRANSPORT = "stdio"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8600
DEFAULT_WORKFLOW_NAME = "krea2_turbo_t2i.json"

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

ENV_TEMPLATE_HINT = (
    "Copy .env.example to .env and fill in RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID, "
    "or set them as environment variables for the MCP client."
)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def mask_secret(value: str | None, *, keep_start: int = 4, keep_end: int = 4) -> str:
    """Render a secret so it can be logged: ``abcd…wxyz``.

    Short values are masked completely; an empty value reports ``<unset>`` so a
    half-configured server is obvious in logs and in ``check_setup``.
    """
    if not value:
        return "<unset>"
    if len(value) <= keep_start + keep_end:
        return "*" * len(value)
    return f"{value[:keep_start]}...{value[-keep_end:]}"


def find_env_file(explicit: str | Path | None = None) -> Path | None:
    """Locate the ``.env`` file to load.

    Order: an explicit ``--env-file``, then the current working directory, then
    the repository root. Returns ``None`` when there is nothing to load (which is
    normal for a published package driven entirely by real environment variables).
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"--env-file {path} does not exist.")
        return path

    seen: list[Path] = []
    for candidate in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate.is_file():
            return candidate
    return None


def load_env(env_file: str | Path | None = None) -> Path | None:
    """Load ``.env`` without overriding variables that are already set."""
    path = find_env_file(env_file)
    if path is not None:
        load_dotenv(path, override=False)
    return path


def default_workflow_path() -> Path:
    """Best guess at the bundled Krea2 Turbo template."""
    candidates = (
        PROJECT_ROOT / "workflows" / DEFAULT_WORKFLOW_NAME,
        Path.cwd() / "workflows" / DEFAULT_WORKFLOW_NAME,
        PACKAGE_ROOT / "workflows" / DEFAULT_WORKFLOW_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration."""

    api_key: str
    endpoint_id: str
    base_url: str = DEFAULT_BASE_URL
    job_timeout_s: float = DEFAULT_JOB_TIMEOUT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    output_dir: Path = field(default_factory=lambda: Path("output"))
    workflow_path: Path = field(default_factory=default_workflow_path)
    transport: str = DEFAULT_TRANSPORT
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    public_base_url: str = ""

    @property
    def endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.endpoint_id}"

    @property
    def masked_api_key(self) -> str:
        return mask_secret(self.api_key)

    @property
    def http_enabled(self) -> bool:
        return self.transport == "streamable-http"

    def media_base_url(self) -> str:
        """Base URL clients can use to fetch rendered files, or ``""`` for stdio."""
        if not self.http_enabled:
            return ""
        if self.public_base_url:
            return self.public_base_url.rstrip("/")
        host = self.host if self.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
        return f"http://{host}:{self.port}"

    def summary(self) -> dict:
        """Config report that is safe to show a user or log."""
        return {
            "runpod_endpoint_id": self.endpoint_id or "<unset>",
            "runpod_api_key": self.masked_api_key,
            "runpod_base_url": self.base_url,
            "endpoint_url": self.endpoint_url if self.endpoint_id else "<unset>",
            "workflow_path": str(self.workflow_path),
            "workflow_exists": self.workflow_path.is_file(),
            "output_dir": str(self.output_dir),
            "job_timeout_s": self.job_timeout_s,
            "poll_interval_s": self.poll_interval_s,
            "transport": self.transport,
            "media_base_url": self.media_base_url() or "<stdio: no HTTP server>",
        }


def _text(env: dict, name: str) -> str:
    return (env.get(name) or "").strip()


def _number(env: dict, name: str, default: float, problems: list[str],
            *, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = _text(env, name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        problems.append(f"{name} must be a number; got {raw!r}.")
        return default
    if minimum is not None and value < minimum:
        problems.append(f"{name} must be at least {minimum}; got {value}.")
    if maximum is not None and value > maximum:
        problems.append(f"{name} must be at most {maximum}; got {value}.")
    return value


def _path(env: dict, name: str, default: Path) -> Path:
    raw = _text(env, name)
    if not raw:
        return default.resolve()
    return Path(raw).expanduser().resolve()


def load_settings(
    env_file: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Read and validate configuration.

    ``environ`` is mainly for tests: when supplied, ``.env`` is not consulted and
    nothing is read from the process environment.

    All problems are collected and reported together so a first-time user fixes
    one screen of issues instead of one at a time.
    """
    if environ is None:
        load_env(env_file)
        env: dict[str, str] = dict(os.environ)
    else:
        env = dict(environ)

    problems: list[str] = []

    api_key = _text(env, "RUNPOD_API_KEY")
    if not api_key:
        problems.append(
            "RUNPOD_API_KEY is not set. Get a key from "
            "https://www.runpod.io/console/user/settings → API Keys."
        )

    endpoint_id = _text(env, "RUNPOD_ENDPOINT_ID")
    if not endpoint_id:
        problems.append(
            "RUNPOD_ENDPOINT_ID is not set. Copy it from the endpoint URL: "
            "https://www.runpod.io/console/serverless → your endpoint "
            "(the id is the part after /v2/)."
        )
    elif endpoint_id.startswith(("http://", "https://")):
        problems.append(
            "RUNPOD_ENDPOINT_ID must be the bare endpoint id, not a URL. "
            "Put the host (https://api.runpod.ai/v2) in RUNPOD_BASE_URL instead."
        )
    elif "/" in endpoint_id or " " in endpoint_id:
        problems.append(f"RUNPOD_ENDPOINT_ID contains invalid characters: {endpoint_id!r}.")

    base_url = _text(env, "RUNPOD_BASE_URL") or DEFAULT_BASE_URL
    if not base_url.startswith(("http://", "https://")):
        problems.append(f"RUNPOD_BASE_URL must start with http:// or https://; got {base_url!r}.")

    job_timeout_s = _number(env, "RUNPOD_JOB_TIMEOUT_S", DEFAULT_JOB_TIMEOUT_S, problems, minimum=10)
    poll_interval_s = _number(env, "RUNPOD_POLL_INTERVAL_S", DEFAULT_POLL_INTERVAL_S, problems, minimum=0.25)

    workflow_path = _path(env, "WORKFLOW_PATH", default_workflow_path())
    if not workflow_path.is_file():
        problems.append(
            f"Workflow template not found: {workflow_path}. "
            "Set WORKFLOW_PATH to a ComfyUI API-format JSON file."
        )

    transport = (_text(env, "MCP_TRANSPORT") or DEFAULT_TRANSPORT).lower()
    if transport not in ("stdio", "streamable-http"):
        problems.append(
            f"MCP_TRANSPORT must be 'stdio' or 'streamable-http'; got {transport!r}."
        )

    port = int(_number(env, "MCP_PORT", DEFAULT_PORT, problems, minimum=1, maximum=65535))

    settings = Settings(
        api_key=api_key,
        endpoint_id=endpoint_id,
        base_url=base_url.rstrip("/"),
        job_timeout_s=job_timeout_s,
        poll_interval_s=poll_interval_s,
        output_dir=_path(env, "RUNPOD_OUTPUT_DIR", Path.cwd() / "output"),
        workflow_path=workflow_path,
        transport=transport,
        host=_text(env, "MCP_HOST") or DEFAULT_HOST,
        port=port,
        public_base_url=_text(env, "MCP_PUBLIC_BASE_URL"),
    )

    if problems:
        bullet = "\n".join(f"  {index}. {problem}" for index, problem in enumerate(problems, 1))
        raise ConfigError(f"Configuration problems:\n{bullet}\n\n{ENV_TEMPLATE_HINT}")

    return settings
