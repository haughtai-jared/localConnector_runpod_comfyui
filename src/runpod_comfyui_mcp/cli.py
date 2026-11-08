"""Command line entry point.

The server never exits because configuration is missing — it starts, logs a loud
warning, and every tool reports the same actionable error. That keeps an MCP host
(which otherwise just sees "server crashed") useful. ``--check`` is the strict
mode for setup and CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace

from . import __version__
from .config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TRANSPORT,
    ConfigError,
    Settings,
    load_env,
    load_settings,
)
from .runpod import RunpodError
from .server import configure, format_config_error, get_client, server
from .workflow import describe_template

logger = logging.getLogger("runpod_comfyui_mcp")

TRANSPORTS = ("stdio", "streamable-http")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runpod-comfyui-mcp",
        description=(
            "Local MCP connector that prepares a ComfyUI workflow and runs it on a "
            "Runpod serverless endpoint."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default=None,
        help="MCP transport. Default: stdio (or MCP_TRANSPORT).",
    )
    parser.add_argument("--host", default=None, help=f"HTTP bind address (default {DEFAULT_HOST}).")
    parser.add_argument("--port", type=int, default=None, help=f"HTTP port (default {DEFAULT_PORT}).")
    parser.add_argument("--env-file", default=None, help="Path to a .env file to load.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and the endpoint, then exit.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def _resolve_options(args: argparse.Namespace) -> tuple[str, str, int]:
    transport = (args.transport or os.environ.get("MCP_TRANSPORT") or DEFAULT_TRANSPORT).lower()
    if transport not in TRANSPORTS:
        raise ConfigError(f"MCP_TRANSPORT must be one of {', '.join(TRANSPORTS)}; got {transport!r}.")
    host = args.host or os.environ.get("MCP_HOST") or DEFAULT_HOST
    raw_port = args.port if args.port is not None else os.environ.get("MCP_PORT")
    try:
        port = int(raw_port) if raw_port not in (None, "") else DEFAULT_PORT
    except ValueError as exc:
        raise ConfigError(f"MCP_PORT must be an integer; got {raw_port!r}.") from exc
    return transport, host, port


def _print_check(settings: Settings, transport: str, host: str, port: int) -> int:
    report: dict = {
        "version": __version__,
        "transport": transport,
        "host": host,
        "port": port,
        "config": settings.summary(),
        "workflow": describe_template(settings.workflow_path),
        "endpoint": {},
    }

    code = 0
    if not report["workflow"].get("valid"):
        code = 2

    try:
        health = get_client().health()
        report["endpoint"] = {"reachable": True, "health": health}
    except RunpodError as exc:
        report["endpoint"] = {"reachable": False, "error": str(exc)}
        code = code or 1

    print(json.dumps(report, indent=2))
    return code


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    # httpx logs every request at INFO, which drowns the connector's own output.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        env_file = load_env(args.env_file)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if env_file is not None:
        logger.debug("Loaded environment from %s", env_file)

    settings: Settings | None = None
    config_error: ConfigError | None = None
    try:
        settings = load_settings(env_file=args.env_file)
    except ConfigError as exc:
        config_error = exc

    try:
        transport, host, port = _resolve_options(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if settings is not None:
        settings = replace(settings, transport=transport, host=host, port=port)
        configure(settings)
        logger.info("Runpod endpoint : %s", settings.endpoint_url)
        logger.info("Workflow        : %s", settings.workflow_path)
        logger.info("Output folder   : %s", settings.output_dir)
    else:
        configure(None, str(config_error))
        print(format_config_error(config_error or ConfigError("unknown error")), file=sys.stderr)

    if args.check:
        if settings is None:
            return 2
        return _print_check(settings, transport, host, port)

    if transport == "stdio":
        logger.info("MCP transport   : stdio")
        server.run(transport="stdio")
        return 0

    logger.info("MCP HTTP        : http://%s:%s/mcp", host, port)
    logger.info("Chat route      : http://%s:%s/v1/chat/completions", host, port)
    logger.info("Usage page      : http://%s:%s/", host, port)
    server.run(
        transport="streamable-http",
        host=host,
        port=port,
        streamable_http_path="/mcp",
        json_response=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
