"""`python capstone.py demo`: the 10-minute four-part standard (W12-Capstone-D.md section 5), in order.
Everything it prints is computed live from the system, except the labelled borrowed figures in cost.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from week12 import agent, evalrun, cost  # noqa: E402
from week12.mcp_host import McpSession  # noqa: E402

FN = {f["fnol_id"]: f for f in json.load(open(ROOT / "week12/data/fnols.json"))["fnols"]}


def h(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def spans(t):
    for s in t["spans"]:
        print(f"   {s['name']:40s} {s['kind']:10s} {s['duration_ms']:8.2f} ms  ${s['cost_usd']:.6f}")


def run():
    h("PART 1 - a real run: FNOL in, referral out (loss 11 Aug 2024, household flood)")
    with McpSession() as sess:
        print("tools discovered at runtime:", json.dumps(sess.raw))
        r = agent.run_triage(FN["F-01"], session=sess)
        print(r["packet"]["text"])
        print("\nvisible tool calls:", " -> ".join(t["tool"] for t in r["trajectory"]))
        spans(r["trace"])
        print(f"trace_id {r['trace']['trace_id']} echoed by the MCP server: "
              f"{all(e['echo_trace_id'] == r['trace']['trace_id'] for e in r['trace']['server_echo'])}")

        h("PART 2 - the failure: the same claim with the date filter OFF (what the old retriever did)")
        bad = agent.run_triage(FN["F-01"], session=sess, enforce_date=False)
        for c in bad["packet"]["citations"]:
            print(f"   cited {c['clause_id']} endorsement {c['endorsement_version']}  (loss date {c['resolved_for_loss_date']})")
        print("   -> a 2024 loss was assessed against wording from the 2025 set; the retriever 'preferred' it.")
        print(f"   trace {bad['trace']['trace_id']}")

    h("PART 3 - a number that moved (all 38 cases, rerun live)")
    evalrun.compare()

    h("PART 4 - what's next: guard case, cost per claim, hailstorm first-break")
    print("permanent guard cases (12 straddle): w12-01 w12-02 w12-05 w12-06 w12-18 w12-20 w12-22 w12-27 w12-31 w12-33 w12-34 w12-37; plus w12-26 w12-35 w12-36 found by the seam/unseen rounds.")
    cost.report()


if __name__ == "__main__":
    run()
