"""OUR MCP server for Week 12: policy-wording search scoped by product AND loss date.

Exposes one genuine capability to any MCP client (ours, or another squad's):

    search_policy_wording(query, product, as_of)

`as_of` (the loss date, YYYY-MM-DD) is REQUIRED and the date filter runs in code
inside the retriever - a client cannot get "the latest wording" by omitting it.

Hops: the host sends params._meta.trace_id on tools/call. This server (a) logs the
call under that trace id and (b) echoes the id and its own server-side timing back in
result._meta, so the trace survives the MCP hop and can be verified from both ends.

PII line: results contain only wording text + clause metadata, and are passed through
pii.redact_obj before leaving the process. The query the client sent is logged
redacted, never verbatim.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from week12 import pii  # noqa: E402
from week12.retriever import get_wordings  # noqa: E402
from week9_mcp.mcp_server import MCPServer, serve  # noqa: E402

LOG = os.environ.get("W12_SERVER_LOG", os.path.join(ROOT, "week12", "runtime", "server_log.jsonl"))


def search_policy_wording(query: str, product: str, as_of: str) -> dict:
    """Search the policy wording IN FORCE on a given date for a product.

    WHEN to call: any time you need the clause text (exclusions, perils, conditions) that
    applies to a claim. HOW: give the peril in policy words (e.g. 'flood exclusion surface
    water'), the product ('household' or 'motor'), and the LOSS DATE as as_of (YYYY-MM-DD).
    The server returns clauses from the endorsement version in force on that date only,
    each with clause_id, endorsement_version and effective dates, so cite all three.
    This tool returns wording; it never states whether a claim is covered.
    """
    if product not in ("household", "motor"):
        return {"error": "product must be 'household' or 'motor'"}
    if not as_of:
        return {"error": "as_of (the loss date, YYYY-MM-DD) is required: the wording depends on the date"}
    try:
        res = get_wordings().retrieve(query, product=product, as_of=as_of, k=3, enforce_date=True,
                                      rerank=os.environ.get("W12_RERANK") == "1")
    except ValueError:
        return {"error": f"as_of {as_of!r} is not an ISO date YYYY-MM-DD"}
    if res["resolved_version"] is None:
        return {"error": f"no endorsement version was in force on {as_of}"}
    return pii.redact_obj({
        "query": query, "product": product, "as_of": as_of,
        "resolved_version": res["resolved_version"], "date_filter": True,
        "hits": [{k: h[k] for k in ("chunk_id", "clause_id", "title", "version", "effective_from",
                                    "effective_to", "text", "score")} for h in res["hits"]],
    })


class TracedServer(MCPServer):
    """MCPServer that reads params._meta.trace_id and echoes it back with server timing."""

    def _handle_initialize(self, params):
        res = super()._handle_initialize(params)
        res["capabilities"]["experimental"] = {"traceEcho": True}   # host may rely on result._meta.trace_id
        return res

    def _handle_tools_call(self, params):
        meta = (params.get("_meta") or {}) if isinstance(params, dict) else {}
        trace_id = meta.get("trace_id")
        t0 = time.perf_counter()
        result = super()._handle_tools_call(params)
        ms = round((time.perf_counter() - t0) * 1000, 3)
        name = params.get("name") if isinstance(params, dict) else None
        args = (params.get("arguments") or {}) if isinstance(params, dict) else {}
        result["_meta"] = {"trace_id": trace_id, "server": self.server_info["name"],
                           "tool": name, "server_ms": ms}
        try:
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            with open(LOG, "a") as fh:
                fh.write(json.dumps({
                    "ts": time.time(), "trace_id": trace_id, "tool": name,
                    "query_redacted": pii.redact(str(args.get("query", ""))),
                    "product": args.get("product"), "as_of": args.get("as_of"),
                    "is_error": result.get("isError"), "server_ms": ms,
                }) + "\n")
        except OSError:
            pass
        return result


def build_server() -> MCPServer:
    s = TracedServer("policy-wording", version="1.0.0")
    s.tool("search_policy_wording", search_policy_wording.__doc__.strip(), {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The peril/clause to find, in policy words."},
            "product": {"type": "string", "enum": ["household", "motor"]},
            "as_of": {"type": "string", "description": "Loss date YYYY-MM-DD. Required."},
        },
        "required": ["query", "product", "as_of"],
    })(search_policy_wording)
    return s


if __name__ == "__main__":
    serve(build_server())
