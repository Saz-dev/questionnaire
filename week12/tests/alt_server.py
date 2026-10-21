"""A THIRD MCP server used only by the tests: same capability, DIFFERENT tool name. If the agent
discovers tools by capability at runtime, it will use this tool when it is the only one registered."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from week12.retriever import get_wordings  # noqa: E402
from week9_mcp.mcp_server import MCPServer, serve  # noqa: E402


def wording_lookup_v2(query: str, product: str, as_of: str) -> dict:
    r = get_wordings().retrieve(query, product=product, as_of=as_of, k=3)
    return {"resolved_version": r["resolved_version"], "date_filter": True,
            "hits": [{k: h[k] for k in ("chunk_id", "clause_id", "title", "version", "effective_from",
                                        "effective_to", "text", "score")} for h in r["hits"]]}


s = MCPServer("alt-wording")
s.tool("wording_lookup_v2", "Find policy wording in force on a date.", {
    "type": "object", "properties": {"query": {"type": "string"}, "product": {"type": "string"}, "as_of": {"type": "string"}},
    "required": ["query", "product", "as_of"]})(wording_lookup_v2)
serve(s)
