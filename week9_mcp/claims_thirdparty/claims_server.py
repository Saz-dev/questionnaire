"""Server TWO — the claims-system MCP server (THIRD PARTY).

Provided by the claims-platform team as a black box: claim status by claim
number, and the adjuster-note history behind it. This directory is a
vendored drop-in of their artifact — it is NOT our code, it was not
reviewed by us, and it runs inside our agent's trust boundary holding a
token that can read adjuster notes on every open claim. See risk_note.md.

Wire-compatible with our host: stdio JSON-RPC, initialize -> tools/list ->
tools/call, using the same shared lib (mcp_server.py). The claims server
exposes only read-only lookups; it cannot change claim state.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claims_data import CLAIM_NUMBER_RE_HINT, CLAIMS  # noqa: E402
from mcp_server import MCPServer, serve  # noqa: E402

CLAIM_NUMBER_RE = re.compile(r"^CLM-\d{4}-\d{5}$")


def _lookup(claim_number: str) -> tuple[dict | None, str | None]:
    """Return (record, none) or (none, recoverable_error)."""
    if not isinstance(claim_number, str) or not CLAIM_NUMBER_RE.match(claim_number):
        return None, (
            f"claim {claim_number!r} is malformed: {CLAIM_NUMBER_RE_HINT}. "
            f"Correct the number and retry."
        )
    record = CLAIMS.get(claim_number)
    if record is None:
        known = ", ".join(sorted(CLAIMS.keys()))
        return None, (
            f"claim {claim_number} not found: {CLAIM_NUMBER_RE_HINT}; claims on "
            f"file: {known}. If you typo'd the claim number, retry with the "
            f"corrected form."
        )
    return record, None


def claim_status(claim_number: str) -> dict:
    record, error = _lookup(claim_number)
    if error:
        return {"error": error}
    return {
        "claim_number": claim_number,
        "claimant": record["claimant"],
        "policy_number": record["policy_number"],
        "status": record["status"],
        "reported": record["reported"],
        "claimed_amount": record["claimed_amount"],
    }


def claim_adjuster_notes(claim_number: str) -> dict:
    record, error = _lookup(claim_number)
    if error:
        return {"error": error}
    return {
        "claim_number": claim_number,
        "notes": record["adjuster_notes"],
    }


def build_server() -> MCPServer:
    s = MCPServer("claims-system", version="3.2.1")
    s.tool(
        "claim_status",
        "Look up the current status of an insurance claim by claim number "
        "(CLM-YYYY-nnnnn): claimant, linked policy, status, reported date and "
        "claimed amount. Read-only.",
        {
            "type": "object",
            "properties": {
                "claim_number": {
                    "type": "string",
                    "description": "The claim number, e.g. 'CLM-2024-88120'.",
                }
            },
            "required": ["claim_number"],
        },
    )(claim_status)
    s.tool(
        "claim_adjuster_notes",
        "Read the adjuster's note history for one insurance claim by claim "
        "number (CLM-YYYY-nnnnn). Sensitive: notes may name claimants and "
        "their address/damage details. Read-only.",
        {
            "type": "object",
            "properties": {
                "claim_number": {
                    "type": "string",
                    "description": "The claim number, e.g. 'CLM-2024-88120'.",
                }
            },
            "required": ["claim_number"],
        },
    )(claim_adjuster_notes)
    return s


if __name__ == "__main__":
    serve(build_server())