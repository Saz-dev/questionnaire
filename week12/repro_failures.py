"""Re-creates, on demand, the failures the integrated eval found BEFORE they were fixed, by running the
current system with the OLD behaviour patched back in. Writes retrievable traces (week12/runtime/
failure_traces.jsonl) so every taxonomy mode has a real trace id that `capstone.py trace <id>` can open.
Old behaviours (each was a real defect, see docs/eval_run_1_red.txt and docs/eval_run_3_unseen_cases.txt):
  seam-intake  : intake passed the loss date / policy number through untouched
  peril        : classify_peril took the first rule that matched, and 'fire' did not match 'bushfire'
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from week12 import agent, evalrun  # noqa: E402
from week12.mcp_host import McpSession  # noqa: E402

OLD_RULES = [(p, rx if p != "fire" else r"\b(fire|chip pan|smoke)\b") for p, rx in agent.PERIL_RULES]


def classify_v1(text):
    for peril, rx in OLD_RULES:
        if re.search(rx, text or "", re.I):
            return peril
    return None


def run() -> list[dict]:
    cases = {c["id"]: c for c in evalrun.load_cases()}
    new = (agent.classify_peril, agent.normalize_loss_date, agent.canon_policy_no)
    out, traces = [], []
    try:
        agent.classify_peril = classify_v1
        agent.normalize_loss_date = lambda raw: (raw, None if raw else "loss_date_missing")
        agent.canon_policy_no = lambda raw: raw
        with McpSession() as s:
            for cid in ("w12-26", "w12-27", "w12-32", "w12-35", "w12-36", "w12-37"):
                c = cases[cid]
                r = agent.run_triage(evalrun.fnol_for(c), session=s, user_id="repro")
                from week12 import assertions
                a = assertions.run_all(r, c)
                traces.append(r["trace"])
                out.append({"case": cid, "mode": c["category"], "trace_id": r["trace"]["trace_id"],
                            "failed_checks": a["failed"], "trajectory": [t["tool"] for t in r["trajectory"]],
                            "expected_trajectory": c["expect"]["trajectory"], "outcome_pass": a["passed"],
                            "peril_got": r["packet"]["peril"], "peril_expected": c["expect"]["peril"],
                            "referral_reason": r["packet"]["referral"]["reason"]})
    finally:
        agent.classify_peril, agent.normalize_loss_date, agent.canon_policy_no = new
    (Path(__file__).parent / "runtime" / "failure_traces.jsonl").write_text(
        "\n".join(json.dumps(t) for t in traces) + "\n")
    return out


if __name__ == "__main__":
    for r in run():
        print(json.dumps(r))
