"""
BONUS — MCP gateway: both servers behind one front door.

The agent connects to ONE server ('mcp-gateway'); the gateway is itself an
MCP server (stdio) that fans out to the real policy-docs and claims-system
servers over MCP, writes one audit line per tools/call, and enforces a
scoped token at the door:

    --token full   claim_status + claim_adjuster_notes + policy tools
    --token ro     claim_status + policy tools, claim_adjuster_notes DENIED

A denied call returns a *recoverable* isError message rather than killing
the conversation: the model reads why it was denied and answers from the
tools it still has (see traces/bonus_demo.json).

Audit line (one per tools/call, all on one line):
    <ts> caller=<clientInfo> scope=<token scope> tool=<name>
         claim=<claim/policy number if present> denied=<0|1>

The gateway never modifies either backend server and does not run any model.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_client import MCPConnection
from mcp_server import MCPServer, serve

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT_FILE = os.path.join(HERE, "gateway_audit.log")

# token scope -> allowed tool names
FULL = {"claim_status", "claim_adjuster_notes", "search_policy_documents", "get_policy"}
READONLY = {"claim_status", "search_policy_documents", "get_policy"}
DENY_TOOL = "claim_adjuster_notes"

BASE_SERVERS = {
    "policy-docs": {"command": "python", "args": ["servers/policy_server.py"]},
    "claims-system": {"command": "python", "args": ["claims_thirdparty/claims_server.py"]},
}


def _audit(entry: dict):
    with open(AUDIT_FILE, "a") as fh:
        line = " ".join(f"{k}={v}" for k, v in entry.items())
        fh.write(line + "\n")


def _claim_ref(tool: str, args: dict) -> str:
    for key in ("claim_number", "policy_number"):
        if key in args:
            return str(args[key])
    if tool == "search_policy_documents":
        return str(args.get("query", ""))[:28]
    return "-"


def build_gateway(token_scope: str) -> MCPServer:
    if token_scope not in ("full", "ro"):
        raise SystemExit(f"unknown token scope {token_scope!r}")
    allowed = FULL if token_scope == "full" else READONLY

    conns = {name: MCPConnection(name, cfg, wire_file=None, cwd=HERE) for name, cfg in BASE_SERVERS.items()}
    for conn in conns.values():
        conn.initialize()

    tool_map = {}
    tool_schemas = []
    for sname, conn in conns.items():
        for t in conn.list_tools():
            tool_map[t["name"]] = (sname, conn)
            tool_schemas.append(t)

    s = MCPServer("mcp-gateway", version="1.0.0")

    def make_handler(name: str):
        def handler(**args):
            is_denied = name not in allowed
            extra = {"host": None}
            _audit({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "caller": "week9-mcp-agent",
                "scope": token_scope,
                "tool": name,
                "claim": _claim_ref(name, args),
                "denied": "1" if is_denied else "0",
            })
            if is_denied:
                return {"error": (
                    f"denied: the scoped token you presented (scope '{token_scope}') "
                    f"does not include {name}. The gateway allows only: "
                    f"{', '.join(sorted(allowed))}. Answer from those tools; if they "
                    f"cannot give the detail, say the caller needs a token with broader "
                    f"scope."
                )}
            sname, conn = tool_map[name]
            return json.loads(conn.call_tool(name, args)["content"][0]["text"])
        return handler

    for t in tool_schemas:
        s.tool(t["name"], t["description"], t["inputSchema"])(make_handler(t["name"]))
    return s


if __name__ == "__main__":
    scope = "ro"
    if "--token" in sys.argv:
        scope = sys.argv[sys.argv.index("--token") + 1]
    serve(build_gateway(scope))