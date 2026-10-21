"""MCP host side for Week 12: reuses the Week-9 stdio client, adds trace propagation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from week9_mcp.mcp_server import PROTOCOL_VERSION  # noqa: E402,F401
import os  # noqa: E402
import queue  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402


class MCPConnection:
    """Stdio JSON-RPC client (same protocol and shape as week9_mcp/mcp_client.py). Copied here, not
    imported, because the Week 9 file uses `X | None` annotations that break on Python 3.9 and the
    Week 9 files are pinned by committed diffs, so they are left byte-identical."""

    def __init__(self, name, config, wire_file=None, cwd=".", timeout_s=30.0):
        self.name, self.timeout_s, self._id, self._closed = name, timeout_s, 0, False
        cmd = sys.executable if config["command"] == "python" else config["command"]
        self.proc = subprocess.Popen([cmd, *config.get("args", [])], cwd=cwd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if line.strip():
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    self._q.put({"id": None})

    def _send(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None):
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                msg = self._q.get(timeout=max(deadline - time.monotonic(), 0.01))
            except queue.Empty:
                raise TimeoutError(f"{self.name}: no reply to {method} within {self.timeout_s}s")
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"{self.name}: {method} error: {msg['error']}")
                return msg.get("result", {})

    def initialize(self):
        res = self.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                          "clientInfo": {"name": "week12-host", "version": "1.0.0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        return res

    def list_tools(self):
        return self.request("tools/list", {"cursor": None}).get("tools", [])

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


class TracedConnection(MCPConnection):
    """Same transport; tools/call carries params._meta.trace_id (MCP's reserved _meta field)."""

    echoes_trace = False

    def initialize(self):
        res = super().initialize()
        self.echoes_trace = bool(res.get("capabilities", {}).get("experimental", {}).get("traceEcho"))
        return res

    def call_tool_traced(self, name: str, arguments: dict, trace_id: str) -> tuple[dict, float]:
        t = time.perf_counter()
        result = self.request("tools/call", {"name": name, "arguments": arguments,
                                             "_meta": {"trace_id": trace_id}})
        return result, (time.perf_counter() - t) * 1000


def discover(config_path: Path | None = None) -> tuple[dict, dict, dict]:
    """Connect to every server in the config and run tools/list. Returns
    (connections, tools_by_name, raw_discovery). No tool name is hardcoded here."""
    cfg = json.loads((config_path or Path(__file__).parent / "mcp_config.json").read_text())
    conns, tools, raw = {}, {}, {}
    for sname, sconf in cfg.get("mcpServers", {}).items():
        conn = TracedConnection(sname, sconf, wire_file=None, cwd=str(ROOT), timeout_s=8.0)
        try:
            conn.initialize()
            listed = conn.list_tools()
        except Exception as exc:  # a server that will not start must not take the triage down with it
            conn.close()
            print(f"[mcp] server {sname!r} skipped: {exc}", file=sys.stderr)
            continue
        conns[sname] = conn
        raw[sname] = [t["name"] for t in listed]
        for t in listed:
            tools[t["name"]] = {"server": sname, "schema": t.get("inputSchema", {}),
                                "description": t.get("description", "")}
    return conns, tools, raw


class McpSession:
    """Connect + discover once, reuse across many triage runs (eval), close at the end."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path
        self.conns: dict = {}
        self.tools: dict = {}
        self.raw: dict = {}

    def __enter__(self):
        self.conns, self.tools, self.raw = discover(self.config_path)
        return self

    def __exit__(self, *exc):
        for c in self.conns.values():
            c.close()
