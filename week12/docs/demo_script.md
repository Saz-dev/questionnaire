<!-- Soft Suave · The AI Engineering League -->
# 10-minute demo script (four-part standard)

Run `python capstone.py demo` for a scripted pass; timings below are for talking over it. Everything printed is computed live.

## Part 1 — a real run (≈3 min)  `python capstone.py triage F-01`
Say: "11 Aug 2024, household flood, policy HH-10001." Point at, in order:
1. tools discovered at runtime over MCP (`policy-wording`, `claims-system`) — not hardcoded;
2. tool calls: `policy_lookup → coverage_in_force_check → search_policy_wording → refer_to_adjuster`;
3. in force on 2024-08-11 ✔ (range shown);
4. citation: `H-3.4 Flood`, **endorsement 2023**, effective 2023-01-01 → 2024-12-31, **resolved for loss date 2024-08-11**, clickable link;
5. three money figures: deductible 1,500 · sub-limit 15,000 · policy limit 250,000;
6. "REFERRED to adjuster-review." Nowhere does it say covered. Spans: cost $0, latency per span; `server echoed trace_id == ours: True`.

## Part 2 — the failures (≈3 min)  `python capstone.py triage F-01 --before`
Same claim, date filter off → cites the **2025** flood exclusion for a 2024 loss (the regulator's finding). Frequency: **22/30 cases** got different wording than was in force; **12/12** straddle cases. Then: the mode integration created — **seam-intake** (3/3 first-contact failures: ISO datetime, DD/MM, lower-case policy number; `322d4cfc…`) — and the trace gap at the third-party hop. Then the weaknesses you wrote on Wednesday (`weaknesses.md`; **put your own here**). Say what the unseen round found: 3 of 4 new cases failed.

## Part 3 — a number that moved (≈2 min)  `python capstone.py eval --compare`
Wrong-wording citation **22/30 → 0/30**; straddle slice **0/12 → 12/12**; PII held-out leak **2 → 0** cases; n = 38. Judge agreement: **not re-measured; Week 6 files recompute to 64%/60%** (say it first). One sentence on what the suite does not measure: *it never calls a model, so it says nothing about LLM behaviour, and the corpus is a 26-chunk synthetic slice.*

## Part 4 — what's next (≈2 min)  `python capstone.py cost`
Guard cases: the 12 straddle cases (w12-01, w12-02, w12-05, w12-06, w12-18, w12-20, w12-22, w12-27, w12-31, w12-33, w12-34, w12-37) + w12-26, w12-35, w12-36 (found by the seam/unseen rounds). Cost/claim: $0 default; ≈$0.0005 with a model (borrowed). Hailstorm 18,000 FNOLs: **TPM limit breaks first** — 7.5/min peak × 2,442 tok = 18,315 TPM vs 8,000 (2.3×); limit hit at 3.3/min; compute has ~5,700× headroom.

## Hostile questions — answers grounded in the repo
1. *11 Aug 2024, you cited 2025 — how many cases straddle, what do they score?* 12 straddle cases; **0/12 before, 12/12 after** (`eval_report.md` §3); they are their own slice, not averaged in. Caveat: fixed after seeing them.
2. *Does it ever say "covered"?* The packet is composed by code; `packet.guard_narrative` replaces decision words; `assertions.never_states_decision` fails the case if any appear outside quoted wording; 0/38 plus injection cases w12-18/28/30/38. Not measured on the `--llm` path.
3. *Sub-limit vs deductible vs limit?* Three fields, asserted distinct and exact by `money_three_distinct`; trace: F-01 → 1500/15000/250000. The failure that *did* occur: w12-36 gave flood figures for a fire (wrong peril), caught by that assertion.
4. *Reproduce claim X in 18 months?* Each trace pins `data_hash`, `wording_version`, `retrieved_context_ids`, `packet_sha`, planner, model, prompt version. `python capstone.py replay <trace_id>` re-runs and compares `packet_sha`; it refuses if the data files changed. Limit: reproducibility of the **LLM path** is not guaranteed (temperature 0 is not determinism).
5. *PII?* Logs: FNOL free text is stored as sha+length only; our MCP server returns clause text and metadata only and logs a *redacted* query (`test_our_mcp_server_returns_and_logs_no_pii`). The third-party server's claimant fields are stripped at our boundary. Honest gap: `claim_adjuster_notes` is never exercised, and `redact()` misses cue-less names (weaknesses).
