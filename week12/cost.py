"""Cost per claim triaged + the 10x / hailstorm plan, with the arithmetic shown.

MEASURED here: latency per triage (rules planner, incl. the MCP hop) from the last `eval` run; $0 cost
on the default path (no model call). NOT measured here (no AI_API_KEY in this checkout): the LLM-planner /
LLM-narrative path. For that path the token figure is BORROWED from Week 11's measured agent runs
(week11/cost_by_stage.md: v2 mean 2442 tokens/query, $0.000524/query) and labelled as such.
"""
from __future__ import annotations

import json
from pathlib import Path

import obs

HERE = Path(__file__).parent

# ---- stated assumptions (edit and re-run) ---------------------------------
HAIL_WEEK_FNOLS = 18_000          # from the brief
SHARE_IN_PEAK_48H = 0.60          # ASSUMPTION: 60% of a hailstorm week's FNOLs arrive in the first 48 h
PEAK_HOUR_FACTOR = 2.0            # ASSUMPTION: busiest hour runs at 2x the 48h average
BASELINE_FNOL_PER_MONTH = 1_800   # from the brief
LLM_TOKENS_PER_CLAIM = 2442       # BORROWED: week11 v2 mean tokens/query (measured on the Week-9 agent, not re-measured)
LLM_COST_PER_CLAIM = 0.000524     # BORROWED: week11/cost_by_stage.md, v2 mean $/query


def report() -> None:
    f = HERE / "runtime" / "eval_after.json"
    if not f.exists():
        print("run `python capstone.py eval` first (it writes the latency this report uses)")
        return
    rows = json.loads(f.read_text())["rows"]
    lat = sorted(r["latency_ms"] for r in rows)
    mean_ms = sum(lat) / len(lat)
    p95 = lat[int(0.95 * (len(lat) - 1))]
    cost = sum(r["cost_usd"] for r in rows) / len(rows)

    print(f"== Cost per claim triaged ==")
    print(f"default path (rules planner, no model call): ${cost:.6f}/claim  (MEASURED, n={len(rows)}); "
          f"latency mean {mean_ms:.1f} ms, p95 {p95:.1f} ms (MEASURED, includes the MCP stdio hop)")
    print(f"LLM planner + narrative path: ~${LLM_COST_PER_CLAIM:.6f}/claim at {LLM_TOKENS_PER_CLAIM} tokens "
          f"(BORROWED from week11, see module docstring) -> {LLM_COST_PER_CLAIM * BASELINE_FNOL_PER_MONTH:.2f} $/month at "
          f"{BASELINE_FNOL_PER_MONTH} FNOL/month; the time saved is ~20 min x {BASELINE_FNOL_PER_MONTH} = "
          f"{BASELINE_FNOL_PER_MONTH * 20 / 60:.0f} adjuster-hours/month, so $ is not the constraint")

    peak_48h_per_min = HAIL_WEEK_FNOLS * SHARE_IN_PEAK_48H / (48 * 60)
    peak_hour_per_min = peak_48h_per_min * PEAK_HOUR_FACTOR
    llm_tpm = peak_hour_per_min * LLM_TOKENS_PER_CLAIM
    max_llm_req_per_min = obs.RATE_LIMIT_TPM / LLM_TOKENS_PER_CLAIM
    single_worker_per_min = 60_000 / mean_ms
    print(f"\n== 10x / hailstorm: {HAIL_WEEK_FNOLS:,} FNOLs in a week (assumed {SHARE_IN_PEAK_48H:.0%} in 48 h, "
          f"peak hour {PEAK_HOUR_FACTOR}x) ==")
    print(f"arrival rate: {peak_48h_per_min:.2f}/min over the 48 h, {peak_hour_per_min:.2f}/min in the peak hour "
          f"(vs {BASELINE_FNOL_PER_MONTH / 30 / 24 / 60:.4f}/min average today -> {peak_hour_per_min / (BASELINE_FNOL_PER_MONTH / 30 / 24 / 60):.0f}x)")
    print(f"1. LLM path: {peak_hour_per_min:.2f}/min x {LLM_TOKENS_PER_CLAIM} tok = {llm_tpm:,.0f} TPM vs provider limit "
          f"{obs.RATE_LIMIT_TPM:,} TPM (obs.RATE_LIMIT_TPM) -> {llm_tpm / obs.RATE_LIMIT_TPM:.1f}x over; the limit is hit at "
          f"{max_llm_req_per_min:.1f} FNOL/min.  ** THIS BREAKS FIRST ** (before cost: ${LLM_COST_PER_CLAIM * HAIL_WEEK_FNOLS:.2f} for the whole week)")
    print(f"2. default (no-model) path: one worker does {single_worker_per_min:,.0f}/min "
          f"({single_worker_per_min / peak_hour_per_min:,.0f}x headroom): compute does NOT break; "
          f"what scales is the adjuster queue: {HAIL_WEEK_FNOLS:,} referrals in a week")
    print("3. consequence: keep the model OFF the hot path in a catastrophe (rules planner, no narrative) and spend "
          "model calls only on the referral summary for flagged claims; that is a config switch (--planner rules).")
    print("\nLimits of this number: peak shape is an assumption; the MCP servers are single local stdio processes "
          "(no network latency, no concurrency measured); LLM tokens are borrowed from Week 11, not re-measured.")


if __name__ == "__main__":
    report()
