"""
Server ONE — the claims-policy document search server (OUR server).

This is the MCP server our Week-9 agent discovers tools from first. It
exposes two read-only capabilities over the claims policy corpus:

    search_policy_documents(query)   full-text search over the policy clauses
    get_policy(policy_number)        clause lookup by policy number

History: this server is the AFTER Task-Set-D-requirement-5 rewrite. The
`search_policy_documents` docstring is now written as a prompt for the
model that reads it (when to call it, how to shape the query), and its
no-match path returns a recoverable error instead of the old generic
"Error 3: lookup failed". error_before_after.md replays the same failing
call against VERSION before-rewrite vs this build. Everything else under
this file (corpus, matcher, get_policy) is byte-for-byte unchanged.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import MCPServer, serve

VERSION = "after-rewrite"

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
    """Return the policy clause(s) that cover the event and limit the caller
    asks about, so the caller can answer a coverage question from policy
    language instead of from memory.

    WHEN to call: any question about what a policy covers, excludes, pays, or
    caps (coverage limits, deductibles, covered perils, excluded events).

    HOW to shape the query: name the covered EVENT and the LIMIT you need,
    using the policy's own words. The policy speaks of water-ingress loss as
    "flood" under Endorsement W, so query terms like "flood", "flood
    insurance limit", or "Endorsement W" are what match — "water damage"
    alone does not appear in the corpus.

    ON a recoverable error: read the error, it names the policy vocabulary;
    re-run ONCE with the corrected query. If even a query that names the
    policy term returns no match, report 'not covered under this policy'
    rather than inventing a figure.
    """
    hits = _find_clauses(query)
    if not hits:
        return {
            "error": (
                f"No policy clause matched {query!r}. The policy frames "
                f"water-ingress loss as FLOODING under Endorsement W: re-run "
                f"search_policy_documents with a query that names the policy "
                f"term, e.g. 'flood insurance limit' or 'flooding covered'. "
                f"If a query naming 'flood' still returns no match, answer "
                f"'not covered under this policy' and do not guess a number."
            )
        }
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
        search_policy_documents.__doc__.strip(),
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
        get_policy.__doc__.strip(),
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