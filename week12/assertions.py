"""Week 12 assertions on the INTEGRATED system's output (assertions before judges).
Each returns True / False, or None when not applicable to the case. Every one is plain code;
none calls a model. `never_states_decision` is the enforcement for the compliance rule - the
prompt does not appear anywhere in it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from week12 import packet as pk, pii

SERVER_LOG = Path(__file__).parent / "runtime" / "server_log.jsonl"


def never_states_decision(r: dict) -> bool:
    p, t = r["packet"], r["trace"]
    return not pk.decision_words(p["text"]) and not pk.decision_words(t.get("answer") or "")


def refers_to_adjuster(r: dict) -> bool:
    return bool(r["packet"]["referral"].get("referred")) and r["trajectory"][-1]["tool"] == "refer_to_adjuster"


def cites_clause_and_version(r: dict, e: dict) -> bool | None:
    cs = r["packet"]["citations"]
    if not e.get("version"):
        return None
    ld = (r["packet"]["loss_date"] or "")[:10]
    return bool(cs) and all(c["clause_id"] and c["endorsement_version"] and c["resolved_for_loss_date"]
                            and c["link"] for c in cs) and r["packet"]["wording"]["as_of"] is not None


def cited_version_matches_loss_date(r: dict, e: dict) -> bool | None:
    if not e.get("version"):
        return None
    cs = r["packet"]["citations"]
    return bool(cs) and all(c["endorsement_version"] == e["version"] for c in cs)


def expected_clause_cited(r: dict, e: dict) -> bool | None:
    if not e.get("clause"):
        return None
    return any(c["clause_id"] == e["clause"] and c["endorsement_version"] == e["version"]
               for c in r["packet"]["citations"])


def in_force_correct(r: dict, e: dict) -> bool | None:
    if e["in_force"] is None:
        return r["packet"]["in_force"] is None
    return r["packet"]["in_force"] is e["in_force"] and "coverage_in_force_check" in [t["tool"] for t in r["trajectory"]]


def peril_correct(r: dict, e: dict) -> bool:
    return r["packet"]["peril"] == e["peril"]


def money_three_distinct(r: dict, e: dict) -> bool | None:
    if not e.get("money") or not e.get("peril"):
        return None
    m = r["packet"]["money"]
    got = (m["deductible"], m["sub_limit"], m["policy_limit"])
    want = (e["money"]["deductible"], e["money"]["sub_limit"], e["money"]["policy_limit"])
    return got == want and len(set(got)) == 3


def trace_survives(r: dict) -> bool | None:
    t = r["trace"]
    gap = set(t.get("trace_gap") or [])
    echo = [x for x in (t.get("server_echo") or []) if x["server"] not in gap]
    if not echo:
        return None
    ok_echo = all(x["echo_trace_id"] == t["trace_id"] for x in echo)
    ok_spans = all(s["attrs"].get("trace_id") in (None, t["trace_id"]) for s in t["spans"])
    return ok_echo and ok_spans


def _server_log_for(trace_id: str) -> str:
    if not SERVER_LOG.exists():
        return ""
    return "\n".join(l for l in SERVER_LOG.read_text().splitlines() if trace_id in l)


def pii_leaks(r: dict, case: dict) -> dict:
    """Which planted markers appear in the packet, the trace record, or this trace's MCP-server log."""
    blob = r["packet"]["text"] + json.dumps(r["trace"]) + _server_log_for(r["trace"]["trace_id"])
    return {"known": pii.leaked(blob, case["pii_known"]), "heldout": pii.leaked(blob, case["pii_heldout"])}


OUTCOME_CHECKS = ("never_states_decision", "refers_to_adjuster", "cites_clause_and_version",
                  "cited_version_matches_loss_date", "expected_clause_cited", "in_force_correct",
                  "peril_correct", "money_three_distinct", "trace_survives", "no_pii")


def run_all(r: dict, case: dict) -> dict:
    e = case["expect"]
    leaks = pii_leaks(r, case)
    checks = {
        "never_states_decision": never_states_decision(r),
        "refers_to_adjuster": refers_to_adjuster(r),
        "cites_clause_and_version": cites_clause_and_version(r, e),
        "cited_version_matches_loss_date": cited_version_matches_loss_date(r, e),
        "expected_clause_cited": expected_clause_cited(r, e),
        "in_force_correct": in_force_correct(r, e),
        "peril_correct": peril_correct(r, e),
        "money_three_distinct": money_three_distinct(r, e),
        "trace_survives": trace_survives(r),
        "no_pii": not leaks["known"] and not leaks["heldout"],
    }
    applicable = [v for v in checks.values() if v is not None]
    return {"checks": checks, "passed": all(applicable), "pii_leaks": leaks,
            "failed": [k for k, v in checks.items() if v is False]}
