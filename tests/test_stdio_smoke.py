"""End-to-end check that the packaged entry point speaks MCP over stdio.

Runs the real subprocess, so it covers argument parsing, logging setup, tool
registration and the stdio transport in one go.

stdin is kept open while responses are collected: closing it immediately races the
server's own shutdown and can truncate the session before ``tools/list`` is answered.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "smoke-test", "version": "1.0"},
    },
}


def _pump(stream, sink: "queue.Queue") -> None:
    for line in stream:
        sink.put(line)
    sink.put(None)


def _mcp_session(messages: list[dict], expected_ids: set[int], timeout: float = 60.0) -> dict[int, dict]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("RUNPOD_")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["MCP_TRANSPORT"] = "stdio"
    env["LOG_LEVEL"] = "ERROR"

    proc = subprocess.Popen(
        [sys.executable, "-m", "runpod_comfyui_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=ROOT,
        env=env,
    )

    responses: dict[int, dict] = {}
    sink: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_pump, args=(proc.stdout, sink), daemon=True)
    reader.start()

    try:
        for message in messages:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()

        deadline = time.monotonic() + timeout
        while expected_ids - set(responses):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line = sink.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                continue
            if line is None:
                break
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and payload.get("id") in expected_ids:
                responses[payload["id"]] = payload
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - would mean the server hung
            proc.kill()
            proc.wait(timeout=5)

    return responses


def test_stdio_server_initializes_and_lists_tools():
    responses = _mcp_session(
        [
            INITIALIZE,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ],
        expected_ids={1, 2},
    )

    assert 1 in responses, responses
    assert responses[1]["result"]["serverInfo"]["name"] == "runpod-comfyui-mcp"

    assert 2 in responses, responses
    names = {tool["name"] for tool in responses[2]["result"]["tools"]}
    assert {"generate_image", "check_setup", "cancel_job"} == names


def test_stdio_server_starts_even_without_credentials():
    """A half-configured server must stay usable so the tools can explain why."""
    responses = _mcp_session([INITIALIZE], expected_ids={1})

    assert 1 in responses, responses
