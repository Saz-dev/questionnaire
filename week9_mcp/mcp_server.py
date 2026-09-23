"""
Minimal Model Context Protocol (MCP) server over the stdio transport.

This is deliberately tiny and dependency-free on purpose: it is the
"socket" that both of our MCP servers plug into, and it speaks enough of
the protocol for a standard host (ours in agent.py, or any third-party
MCP client) to talk to it:

    initialize               -> capabilities + serverInfo
    notifications/initialized  (notification, no reply)
    tools/list               -> the tools this server exposes
    tools/call               -> execute one tool by name
    ping                     -> {}

Wire format: JSON-RPC 2.0, newline-delimited JSON on stdin/stdout (the
MCP stdio transport). stdout is the wire and must never carry anything
else — every "tool result" we produce here is a plain data structure, no
model sits anywhere in this process.
"""

import json
import sys


PROTOCOL_VERSION = "2025-03-26"


class Tool:
    """One discoverable tool: name + docstring + JSON schema + handler."""

    def __init__(self, name, description, input_schema, handler):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler


class MCPServer:
    def __init__(self, name: str, version: str = "1.0.0"):
        self.server_info = {"name": name, "version": version}
        self.tools: dict[str, Tool] = {}
        self._next_id = 0

    def tool(self, name: str, description: str, input_schema: dict):
        """Decorator-free registration: attach a handler function."""

        def register(fn):
            self.tools[name] = Tool(name, description, input_schema, fn)
            return fn

        return register

    # ------------------------------------------------------------------
    # JSON-RPC plumbing
    # ------------------------------------------------------------------

    def _read_message(self) -> dict:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        return json.loads(line)

    def _send(self, message: dict):
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()

    def _reply(self, msg_id, result=None, error=None):
        self._send({"jsonrpc": "2.0", "id": msg_id, **(
            {"result": result} if result is not None else {"error": error}
        )})

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "logging": {},
            },
            "serverInfo": self.server_info,
        }

    def _handle_tools_list(self, params: dict) -> dict:
        return {"tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self.tools.values()
        ]}

    def _handle_tools_call(self, params: dict) -> dict:
        if not isinstance(params, dict):
            return self._error_result("tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments") or {}
        tool = self.tools.get(name)
        if tool is None:
            return self._error_result(f"unknown tool: {name!r}")
        if not isinstance(arguments, dict):
            return self._error_result("tool arguments must be an object")
        try:
            payload = tool.handler(**arguments)
        except TypeError as exc:
            # A handler raised because the model passed an argument the
            # schema does not declare. Keep it recoverable: name the error
            # back to the caller instead of killing the server.
            return self._error_result(f"bad argument for {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - recoverable, message to the host
            return self._error_result(f"{type(exc).__name__}: {exc}")
        # A handler that returns {"error": ...} is a recoverable TOOL failure,
        # not a successful lookup — mark it isError so the host can tell a
        # typo'd input apart from a real result.
        is_error = isinstance(payload, dict) and "error" in payload
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": is_error,
        }

    @staticmethod
    def _error_result(message: str) -> dict:
        return {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            try:
                msg = self._read_message()
            except (EOFError, json.JSONDecodeError) as exc:
                print(f"server {self.server_info['name']}: stop ({exc})", file=sys.stderr)
                return

            method = msg.get("method")
            msg_id = msg.get("id")

            if method == "initialize":
                self._reply(msg_id, result=self._handle_initialize(msg.get("params")))
            elif method == "notifications/initialized":
                pass  # notification - no reply expected
            elif method == "ping":
                self._reply(msg_id, result={})
            elif method == "tools/list":
                self._reply(msg_id, result=self._handle_tools_list(msg.get("params")))
            elif method == "tools/call":
                self._reply(msg_id, result=self._handle_tools_call(msg.get("params")))
            else:
                self._reply(msg_id, error={
                    "code": -32601,
                    "message": f"method not found: {method}",
                })


def serve(server: MCPServer):
    server.run()


if __name__ == "__main__":
    s = MCPServer("smoke")
    s.tool(
        "echo",
        "Return the message you are given.",
        {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
    )(lambda message: {"message": message})
    serve(s)