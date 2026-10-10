"""
Week 11 drill — the SQUADMATE runs this, not the person being timed.

Plants ONE bad coverage answer ("covered", exclusion ignored) into the trace log,
among N ordinary traces from "yesterday" (some of them deliberately look similar:
legit "covered" answers, other users, other versions). Prints nothing that
identifies the plant to the finder; the answer key goes to a sealed file.

    python plant_bad_answer.py                 # plant #1, yesterday
    python plant_bad_answer.py --round 2       # bonus: plant a second one
    python plant_bad_answer.py --reveal        # squadmate only: print the key

Filler traces are synthetic (clearly: they are drill data, not real adjuster
traffic) but use the exact v2 schema, written through obs.write_trace_v2, so the
index treats them like any other row.
"""
import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import obs

SEALED = Path(__file__).parent / "week11" / ".drill_sealed.json"
USERS = [f"adj-{i:02d}" for i in range(1, 13)]
MODEL = "openai/gpt-oss-120b"
PV = "agent-v1-2026-10-09"

OK_TEMPLATES = [
    ("What is the status of claim CLM-2024-88120 and what do the adjuster's notes say?",
     "Claim is in_review; notes mention a burst pipe causing flooding and a policy limit check.",
     ["claim:CLM-2024-88120"], ["claim_status", "claim_adjuster_notes"]),
    ("What is the flood coverage limit under Endorsement W?",
     "Flood loss is covered up to $10,000 per occurrence with a $500 deductible.",
     ["clause:flood"], ["search_policy_documents"]),
    ("Why was claim CLM-2024-88123 denied?",
     "Denied: the renter contents policy excludes flooding (rising groundwater).",
     ["claim:CLM-2024-88123", "policy:POL-2024-88123"], ["claim_status", "get_policy"]),
    ("Is earthquake covered under the base policy?",
     "No. Earth movement is excluded from the base policy; a separate endorsement is required.",
     ["clause:earthquake"], ["search_policy_documents"]),
    ("What is the claimed amount for claim CLM-2024-88114?",
     "The claimed amount is $3,400 and the claim is approved.",
     ["claim:CLM-2024-88114"], ["claim_status"]),
    ("Is fire covered under POL-2024-88114?",
     "Yes, fire and lightning are covered under POL-2024-88114.",
     ["policy:POL-2024-88114"], ["get_policy"]),            # legit "covered" decoy
    ("Is windstorm damage to the roof covered?",
     "Yes, windstorm and hail damage to the roof is covered within the base dwelling limit.",
     ["clause:wind"], ["search_policy_documents"]),          # legit "covered" decoy
    ("What is the contents theft deductible?",
     "The standard deductible is $500; theft of contents is payable up to $25,000.",
     ["clause:theft"], ["search_policy_documents"]),
]


def _spans(rng: random.Random, tools: list[str], costly: bool = False) -> list[dict]:
    rec = obs.TraceRecorder()
    laps = len(tools) + 1
    for i in range(laps):
        tin = rng.randint(650, 900) + (rng.randint(0, 40) * i) + (3000 if costly else 0)
        tout = rng.randint(30, 140)
        rec.add_span(f"generation.lap{i+1}", "generation", rng.uniform(450, 1300),
                     tokens_in=tin, tokens_out=tout, cost_usd=obs.llm_cost(MODEL, tin, tout), lap=i + 1)
        if i < len(tools):
            rec.add_span(f"tool.{tools[i]}", "tools", rng.uniform(0.4, 3.0), cost_usd=0.0)
    return rec.spans


def _write(rng, ts, user, q, a, ctx, tools, pv=PV, costly=False):
    t = obs.build_trace(
        question=q, answer=a, prompt_version=pv, model=MODEL, spans=_spans(rng, tools, costly),
        retrieved_context_ids=ctx, source="agent", user_id=user, tools_called=tools,
        ts=ts.isoformat(), extra={"run_id": uuid.uuid4().hex[:8]},
    )
    obs.write_trace_v2(t)
    return t


def plant(round_no: int, n_filler: int, seed: int | None) -> None:
    rng = random.Random(seed)
    day = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def rand_ts():
        return day + timedelta(seconds=rng.randint(8 * 3600, 19 * 3600))

    for _ in range(n_filler):
        q, a, ctx, tools = rng.choice(OK_TEMPLATES)
        _write(rng, rand_ts(), rng.choice(USERS), q, a, ctx, tools, costly=rng.random() < 0.04)

    bad_variants = [
        ("Claim CLM-2024-88123 - is the flood loss in the cellar covered? One line for the claimant.",
         "D. Rose: flood loss is covered (up to $10,000 per occurrence with a $500 deductible).",
         ["claim:CLM-2024-88123", "clause:flood"], ["claim_status", "search_policy_documents"]),
        ("Claimant says rising groundwater flooded the cellar, is that covered? Short answer please.",
         "Yes - rising-groundwater flooding is covered under Endorsement W up to $10,000.",
         ["clause:flood"], ["search_policy_documents"]),
    ]
    q, a, ctx, tools = bad_variants[(round_no - 1) % len(bad_variants)]
    t = _write(rng, rand_ts(), rng.choice(USERS), q, a, ctx, tools)

    sealed = json.loads(SEALED.read_text()) if SEALED.exists() else {}
    sealed[f"round{round_no}"] = {"trace_id": t["trace_id"], "user_id": t["user_id"], "timestamp": t["timestamp"],
                                  "planted_at": datetime.now(timezone.utc).isoformat()}
    SEALED.write_text(json.dumps(sealed, indent=2))
    print(f"planted round {round_no}: {n_filler} filler traces + 1 bad answer written to the log (key sealed).")
    print("Tell the finder ONLY: 'an adjuster says it told a claimant something was covered when the policy excludes it, sometime yesterday.'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--filler", type=int, default=45)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()
    if args.reveal:
        print(SEALED.read_text() if SEALED.exists() else "{}")
    else:
        plant(args.round, args.filler, args.seed)
