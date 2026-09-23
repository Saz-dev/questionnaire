"""
Server ONE — the claims-policy document search server (OUR server).

This is the MCP server our Week-9 agent discovers tools from first. It
exposes two read-only capabilities over the claims policy corpus:

    search_policy_documents(query)   full-text search over the policy clauses
    get_policy(policy_number)        clause lookup by policy number

History: this is the ORIGINAL (pre-rewrite) build. `search_policy_documents`
carries the vague one-line docstring and the generic "Error 3: lookup
failed" no-match error that Task Set D requirement 5 asks us to rewrite —
the rewrite is applied later in the week9_mcp history and evidenced by
error_before_after.md. VERSION marks which build is on the wire.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import MCPServer, serve

VERSION = "before-rewrite"

# ---------------------------------------------------------------------------
# Policy corpus — a small claims policy manual so the server is deterministic
# and self-contained (the real deployment would point this at the RAG index).
# ---------------------------------------------------------------------------

POLICY_CLAUSES = [
    (
        "flood",
        "FLOOD INSURANCE - Under Endorsement W, loss caused by flooding, or by "
        "water entering the dwelling from a storm, a burst pipe, or rising "
        "groundwater, is covered up to a limit of $10,000 per occurrence. A "
        "separate deductible of $500 applies.",
    ),
    (
        "wind",
        "WIND & HAIL - Windstorm and hail damage to the dwelling roof is "
        "covered within the base dwelling limit; no separate endorsement is "
        "required.",
    ),
    (
        "earthquake",
        "EARTHQUAKE - Earth movement is excluded from the base policy; a "
        "separate earthquake endorsement is required for coverage.",
    ),
    (
        "theft",
        "CONTENTS THEFT - Loss of contents by theft is payable up to $25,000, "
        "subject to the standard $500 deductible.",
    ),
    (
        "liability",
        "LIABILITY - Bodily injury liability pays legal defence and judgments "
        "against the insured up to $300,000 per occurrence.",
    ),
]

# policy_number -> (product, cover summary)
POLICIES = {
    "POL-2024-88120": {
        "product": "Dwelling Comprehensive",
        "holder": "A. Marchetti",
        "cover": "Dwelling structure + Endorsement W flood rider",
        "premium": "annual",
    },
    "POL-2024-88114": {
        "product": "Dwelling Fire Only",
        "holder": "K. Obona",
        "cover": "Fire and lightning, no water endorsement",
        "premium": "annual",
    },
    "POL-2024-88123": {
        "product": "Renter Contents",
        "holder": "D. Rose",
        "cover": "Contents only; excludes flooding",
        "premium": "monthly",
    },
}

POLICY_NUMBER_RE = re.compile(r"^POL-\d{4}-\d{5}$")

_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "and", "or", "in", "on", "under",
    "what", "is", "are", "does", "do", "my", "i", "we", "you",
}


def _query_terms(query: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS]


def _find_clauses(query: str) -> list[tuple[str, str]]:
    terms = _query_terms(query)
    if not terms:
        return []
    hits = []
    for key, text in POLICY_CLAUSES:
        lower = text.lower()
        if all(re.search(rf"\b{re.escape(t)}\b", lower) for t in terms):
            hits.append((key, text))
    return hits


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def search_policy_documents(query: str) -> dict:
    """Search the claims policy documents."""
    hits = _find_clauses(query)
    if not hits:
        return {"error": "Error 3: lookup failed"}
    return {
        "query": query,
        "matches": [{"clause": key, "text": text} for key, text in hits],
    }


def get_policy(policy_number: str) -> dict:
    """Return the policy record for one policy number, POL-YYYY-nnnnn."""
    policy = POLICIES.get(policy_number)
    if policy is None:
        return {
            "error": (
                f"policy {policy_number!r} not found: policy numbers look like "
                f"POL-YYYY-nnnnn (e.g. POL-2024-88120). If you typo'd the number, "
                f"retry with the corrected form; if the number is right, there is "
                f"no such policy on file."
            )
        }
    return {"policy_number": policy_number, **policy}


def build_server() -> MCPServer:
    s = MCPServer("policy-docs", version="1.0.0")
    s.tool(
        "search_policy_documents",
        "Search the claims policy documents.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up, e.g. the covered event or limit.",
                }
            },
            "required": ["query"],
        },
    )(search_policy_documents)
    s.tool(
        "get_policy",
        "Return the policy record for one policy number, POL-YYYY-nnnnn.",
        {
            "type": "object",
            "properties": {
                "policy_number": {
                    "type": "string",
                    "description": "The policy number, e.g. 'POL-2024-88120'.",
                }
            },
            "required": ["policy_number"],
        },
    )(get_policy)
    return s


if __name__ == "__main__":
    serve(build_server())