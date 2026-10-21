<!-- Soft Suave · The AI Engineering League -->
# Seam list

> Honest provenance: this list was written from the build log of the session that produced the system, in the order each seam
> was hit. It was **not** kept live on a Tuesday and it has not been edited for flattery; the status column was set at the end.
> Mark any seam you disagree with.

| # | When hit | Seam (what met what) | What broke | Status |
|---|---|---|---|---|
| 1 | first command | repo `.venv` is Python 3.9; Week 9/11 code uses `X \| None` annotations (client + vendored claims server) | import errors / server fails to start | **fixed** — own stdio client in `week12/mcp_host.py`; vendored server launched via `servers/run_vendored.py` (forces postponed annotations, file unmodified); a server that fails to start is skipped, not fatal |
| 2 | design | Week 11 `obs.write_trace_v2` always indexes into the default `traces.db` (path not injectable) | 38-case eval runs would pollute the real log | **fixed** — eval writes its own JSONL under `week12/runtime/`; only `triage` writes the shared log |
| 3 | design | `week9_mcp/mcp_client.call_tool` sends no `_meta`; `MCPServer` hides `params._meta` from handlers | trace id cannot cross the MCP hop | **fixed** — subclassed both in `week12/` (Week 9 files stay byte-identical so their pinned diffs remain valid) |
| 4 | design | Week 9/11 agent prompt *instructs* the model to answer "covered / not covered"; `assertions.exclusion_respected` *requires* it | opposite of the compliance rule | **fixed for Week 12 by replacement, not edit** — capstone path never asks a model for a decision; old Week 9/11 artifacts left as history. Their suite is still the Week 11 record |
| 5 | first eval | `trace_survives` red on the claims-system hop (w12-20, w12-21) | third-party server cannot echo the trace id | **accepted** — flagged `trace_gap:claims-system` on every trace; client-side span keeps cost/latency inside our trace. Server-side correlation not possible without editing vendor code |
| 6 | first eval | FNOL intake sends `2024-08-11T14:30:00` (w12-27) | `coverage_in_force_check` rejected a datetime; packet said "routine referral" with no wording | **fixed** — `normalize_loss_date` |
| 7 | first eval | FNOL intake sends `11/08/2024` (w12-26) | tool called with a bad date; outcome assertions *passed* but trajectory was wrong; referral reason said "routine" | **fixed** — ambiguous dates are never guessed; flagged and referred |
| 8 | first eval | peril lexicon: "Bushfire" did not match `fire` (w12-32) | no peril → no wording search | **fixed** — `\w*fire` |
| 9 | unseen-cases round | `classify_peril` took the first matching rule (w12-35, w12-36) | "fire … hoses flooded" → flood; "storm … water came through" → flood → **wrong deductible and sub-limit** | **fixed** — count-based scoring; still a lexical classifier (see weaknesses) |
| 10 | unseen-cases round | policy number typed `hh-10001` (w12-37) | "policy not found" referral | **fixed** — `canon_policy_no` |
| 11 | obs | `obs.redact_pii` is regex-only (initial+surname, labels); full names / addresses / injuries pass through it | PII in logs | **fixed by not logging** — FNOL free text is stored as sha+len; `week12/pii.py` (registry + patterns) guards everything else |
| 12 | judge | Week 6 judge agreement recomputes to 64% (v1) / 60% (v2), not the documented 80→84% | the ≥85% gate is not met | **ran-out-of-time / blocked** — needs `AI_API_KEY` and a human relabel; see `eval_report.md` §5 |
| 13 | corpus | brief's 500 claims / 60-page wordings / real in-force table do not exist in the repo | the suite runs on a synthetic slice | **accepted** — stated as non-coverage |
| 14 | fresh eyes | Thursday run by another squad | — | **ran-out-of-time** — not done; template in `fresh_eyes.md` |
