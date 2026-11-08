"""End-to-end check that the packaged entry point speaks MCP over stdio.

Runs the real subprocess, so it covers argument parsing, logging setup, tool
registration and the stdio transport in one go.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_stdio(payload: str) -> list[dict]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("RUNPOD_")}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["MCP_TRANSPORT"] = "stdio"
    env["LOG_LEVEL"] = "ERROR"

    completed = subprocess.run(
        [sys.executable, "-m", "runpod_comfyui_mcp"],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        cwd=ROOT,
        env=env,
    )

    messages = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            messages.append(json.loads(line))
        except ValueError:
            continue
    return messages


def test_stdio_server_initializes_and_lists_tools():
    payload = "\n".join(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "smoke-test", "version": "1.0"},
                    },
                }
            ),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        ]
    ) + "\n"

    messages = _run_stdio(payload)
    by_id = {message.get("id"): message for message in messages if "id" in message}

    assert 1 in by_id, messages
    server_info = by_id[1]["result"]["serverInfo"]
    assert server_info["name"] == "runpod-comfyui-mcp"

    assert 2 in by_id, messages
    names = {tool["name"] for tool in by_id[2]["result"]["tools"]}
    assert {"generate_image", "check_setup", "cancel_job"} == names


def test_stdio_server_starts_even_without_credentials():
    """A half-configured server must stay usable so the tools can explain why."""
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "smoke-test", "version": "1.0"},
            },
        }
    ) + "\n"

    messages = _run_stdio(payload)
    assert any(message.get("id") == 1 for message in messages)
