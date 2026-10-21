"""The triage packet and the output guard.

The packet is COMPOSED BY CODE from tool results. A model may only contribute an
optional narrative sentence, and that sentence goes through `guard_narrative`.
So "never state a coverage decision" is enforced here and re-checked by an eval
assertion (assertions.never_states_decision) - not merely asked for in a prompt.

Convention: policy wording is quoted verbatim on lines starting with '> '. Wording
legitimately contains words like "insure"/"covered"; the decision-word ban applies
to everything OUTSIDE those quoted lines.
"""
from __future__ import annotations

import hashlib
import json
import re

DECISION_RE = re.compile(
    r"\b(covered|uncovered|not covered|is excluded|are excluded|will be paid|we will pay|"
    r"will pay|approved|approve|declined|denied|eligible|entitled|liable|payable)\b", re.I)

SAFE_NARRATIVE = ("Facts and wording for the adjuster are listed below; "
                  "no coverage decision has been made by this system.")


def strip_quoted(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(">"))


def decision_words(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in DECISION_RE.finditer(strip_quoted(text or ""))})


def guard_narrative(text: str | None, fallback: str) -> tuple[str, bool]:
    """Return (safe_text, triggered). Any decision word outside quoted wording -> fallback."""
    if not text or not text.strip():
        return fallback, False
    if decision_words(text):
        return fallback, True
    return text.strip(), False


def money_block(limits: dict | None, peril: str | None) -> dict:
    """Deductible, sub-limit and policy limit as THREE separate fields."""
    if not limits or not peril:
        return {"deductible": None, "sub_limit": None, "policy_limit": (limits or {}).get("policy_limit"),
                "applies_to": peril}
    ded = limits["deductible"].get(peril, limits["deductible"]["default"])
    return {"deductible": ded, "sub_limit": limits["sub_limit"].get(peril),
            "policy_limit": limits["policy_limit"], "applies_to": peril}


def build_narrative(p: dict) -> str:
    bits = []
    if p["in_force"] is True:
        bits.append(f"Policy {p['policy_no']} was in force on the loss date {p['loss_date']}.")
    elif p["in_force"] is False:
        bits.append(f"Policy {p['policy_no']} was NOT in force on the loss date {p['loss_date']}.")
    else:
        bits.append("Policy status on the loss date could not be established.")
    if p["peril"]:
        bits.append(f"The loss was classified as: {p['peril'].replace('_', ' ')}.")
    w = p["wording"]
    if w.get("resolved_version"):
        ids = ", ".join(c["clause_id"] for c in p["citations"][:3]) or "none"
        bits.append(f"Wording resolved for loss date {p['loss_date']}: endorsement {w['resolved_version']} "
                    f"(clauses cited: {ids}).")
    bits.append("The claim is referred to an adjuster, who decides.")
    return " ".join(bits)


def render_text(p: dict) -> str:
    L = [f"TRIAGE {p['fnol_id']}   trace_id={p['trace_id']}", "", p["narrative"], ""]
    m = p["money"]
    L += ["Money (three distinct figures):",
          f"  deductible   : {m['deductible']}", f"  sub-limit    : {m['sub_limit']}",
          f"  policy limit : {m['policy_limit']}   (peril: {m['applies_to']})", ""]
    if p["citations"]:
        L.append("Wording cited:")
        for c in p["citations"]:
            L.append(f"  [{c['clause_id']} {c['title']}] endorsement {c['endorsement_version']} "
                     f"(effective {c['effective_from']} to {c['effective_to'] or 'open'}), "
                     f"resolved for loss date {c['resolved_for_loss_date']}  <{c['link']}>")
            L.append(f"  > {c['text']}")
    if p["flags"]:
        L += ["", "Flags: " + ", ".join(p["flags"])]
    r = p["referral"]
    L += ["", f"REFERRED to {r['queue']}: {r['reason']}"]
    return "\n".join(L)


def packet_sha(p: dict) -> str:
    """Hash of the decision-relevant content (excludes trace id and timings) so a replay can prove equality."""
    core = {k: p[k] for k in ("fnol_id", "policy_no", "loss_date", "peril", "in_force", "money",
                              "wording", "referral")}
    core["citations"] = [(c["chunk_id"], c["endorsement_version"], c["text"]) for c in p["citations"]]
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]
