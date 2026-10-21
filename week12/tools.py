"""Local agent tools: policy lookup, coverage-in-force check, refer-to-adjuster.

Deterministic, no model, no network. Each returns a plain dict; a dict with an
"error" key is a recoverable failure (same convention as the Week-9 servers).
These handle ONLY facts. None of them states a coverage decision.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

DATA = Path(__file__).parent / "data"
_POLICIES = {p["policy_no"]: p for p in json.loads((DATA / "policies.json").read_text())["policies"]}
_LIMITS = json.loads((DATA / "limits.json").read_text())


def policy_lookup(policy_no: str) -> dict:
    """Look up a policy by number: product, in-force ranges and the three money
    tables (deductible, sub-limits, policy limit). Never returns holder name or address."""
    p = _POLICIES.get(policy_no)
    if p is None:
        return {"error": f"policy {policy_no!r} not found; refer to an adjuster to locate the policy"}
    lim = _LIMITS[p["product"]]
    return {
        "policy_no": policy_no, "product": p["product"], "in_force_ranges": p["in_force"],
        "limits": {"deductible": lim["deductible"], "sub_limit": lim["sub_limit"],
                   "policy_limit": lim["policy_limit"]},
    }


def coverage_in_force_check(policy_no: str, loss_date: str) -> dict:
    """Was the policy in force on the loss date (YYYY-MM-DD)? Returns the matching range
    or the ranges that were not matched. This is a date fact, not a coverage decision."""
    p = _POLICIES.get(policy_no)
    if p is None:
        return {"error": f"policy {policy_no!r} not found"}
    try:
        d = date.fromisoformat(loss_date)
    except (TypeError, ValueError):
        return {"error": f"loss_date {loss_date!r} is not an ISO date YYYY-MM-DD"}
    for lo, hi in p["in_force"]:
        if date.fromisoformat(lo) <= d <= date.fromisoformat(hi):
            return {"policy_no": policy_no, "loss_date": loss_date, "in_force": True, "range": [lo, hi]}
    return {"policy_no": policy_no, "loss_date": loss_date, "in_force": False, "ranges_on_file": p["in_force"]}


def refer_to_adjuster(reason: str, evidence_ids: list | None = None) -> dict:
    """Hand the claim to a human adjuster with the reason and the evidence ids. The ONLY
    terminal action: the system triages and cites, the adjuster decides."""
    return {"referred": True, "queue": "adjuster-review", "reason": reason,
            "evidence_ids": list(evidence_ids or [])}


LOCAL_TOOLS = {
    "policy_lookup": (policy_lookup, {"type": "object", "properties": {"policy_no": {"type": "string"}}, "required": ["policy_no"]}),
    "coverage_in_force_check": (coverage_in_force_check, {"type": "object", "properties": {"policy_no": {"type": "string"}, "loss_date": {"type": "string"}}, "required": ["policy_no", "loss_date"]}),
    "refer_to_adjuster": (refer_to_adjuster, {"type": "object", "properties": {"reason": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["reason"]}),
}
