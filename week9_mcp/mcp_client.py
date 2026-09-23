"""
MCP client side of the agent module: a stdio JSON-RPC transport that the
host (agent.py) uses to talk to every server listed in mcp_config.json.

Two jobs:
  1. Speak the protocol: initialize -> notifications/initialized ->
     tools/list -> tools/call, newline-delimited JSON on stdin/stdout.
  2. Wire capture: every raw line that crosses the transport is appended
     verbatim to the run's wire/<run_id>__<server>.jsonl so the
     initialize/tools-list/tools-call exchange can be handed to the
     grader unedited (see wire/ and wire.json).

The client is generic hardware: it knows nothing about claims, policies,
or any specific server — only the tools each server advertises.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

PROTOCOL_VERSION = "2025-03-26"


def _resolve_command(command: str) -> str:
    """'python' in a config block means 'the interpreter we are running under'."""
    if command == "python":
        return sys.executable
    return command


class MCPConnection:
    def __init__(self, name: str, config: dict, wire_file: str | None, cwd: str, timeout_s: float = 30.0):
        self.name = name
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.wire_file = wire_file
        command = _resolve_command(config["command"])
        args = config.get("args", [])
        self.proc = subprocess.Popen(
            [command, *args],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._msg_q: queue.Queue = queue.Queue()
        self._closed = False
        self._id = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    # ------------------------------------------------------------------
    # Wire I/O
    # ------------------------------------------------------------------

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            self._log(async_message="read", ts=time.time(), raw=line + "\n")
            try:
                self._msg_q.put(json.loads(line))
            except json.JSONDecodeError:
                self._msg_q.put({"jsonrpc": "2.0", "id": None, "_parse_error": line})

    def _log(self, **fields):
        if not self.wire_file:
            return
        entry = dict(fields)
        with open(self.wire_file, "a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _send(self, payload: dict):
        raw = json.dumps(payload) + "\n"
        self._log(async_message="write", ts=time.time(), raw=raw)
        self.proc.stdin.write(raw)
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        req_id = self._id
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.name}: no reply to {method} within {self.timeout_s}s")
            try:
                msg = self._msg_q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"{self.name}: no reply to {method} within {self.timeout_s}s")
            if isinstance(msg.get("id"), int) and msg.get("id") == req_id:
                if "error" in msg:
                    raise RuntimeError(f"{self.name}: {method} error: {msg['error']}")
                return msg.get("result", {})

    def notify(self, method: str, params: dict | None = None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ------------------------------------------------------------------
    # Protocol helpers
    # ------------------------------------------------------------------

    def initialize(self) -> dict:
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "week9-mcp-agent", "version": "1.0.0"},
        })
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        result = self.request("tools/list", {"cursor": None})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()