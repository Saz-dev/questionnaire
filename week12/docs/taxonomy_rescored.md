<!-- Soft Suave · The AI Engineering League -->
# Week 5 taxonomy, re-scored on the INTEGRATED system

Sample: all 38 cases in `week12/eval_cases.jsonl` (not a random draw of 500 claims: that corpus does not exist here).
Two states are scored, both reproducible:
* **A. integration baseline** = date filter off + FNOL text logged redacted (`python capstone.py eval --before`) — what a naive wire-up gives.
* **B. first contact with the fixes' targets** = the defects the suite found in the first-build code; counted from `eval_run_1_red.txt` and
  `eval_run_3_unseen_cases.txt`, re-creatable with `python week12/repro_failures.py` (it patches the old behaviour back in and writes real traces).

| # | Mode | Freq | Severity | Trace id (open with `python capstone.py trace <id>`) |
|---|---|---|---|---|
| 1 | **wrong-endorsement-version** — loss dated under the 2023 set assessed against 2025 wording because it ranked first | state A: **22/30 cases cite different WORDING than the policy carried that day (73%)**; 12/12 straddle-flood cases wrong | **Critical** (the regulator's finding) | `7064be9e-645d-4bd8-89d5-023c2af3f797` (w12-01, 11 Aug 2024; `docs/evidence_traces.jsonl`) |
| 2 | **peril-misclass → wrong money** — classifier picks the wrong peril, so deductible and sub-limit belong to a different peril | state B: **3/38 (7.9%)**: w12-32, w12-35, w12-36 | High: wrong numbers, plausible-looking | `813a0d15-5d84-4ab9-8380-a83daabc43fa` (w12-36: fire, answered with flood figures; `docs/evidence_traces.jsonl`) |
| 3 | **seam-intake — NEW, created by integration** — an input the FNOL intake accepts but the downstream tool contract rejects or mis-handles (datetime, `DD/MM/YYYY`, lower-case policy no.) and the packet still says "routine referral" | state B: **3/38 (7.9%)**: w12-26, w12-27, w12-37 — **3 of 3 failed first contact** | High: silently skips the in-force check or the wording | `322d4cfc-147f-458d-91b4-547824e87528` (w12-27); `c16096c8-9b16-498e-8028-7d801840b9b1` (w12-26, outcome *passed*) |
| 4 | **pii-in-logs** — free-text FNOL written to traces after regex redaction; cue-less names / hospital names survive | state A: **2/38 cases, 2 of 3 held-out markers leaked** ("Tomasz", "Calder General") | **Critical** | `53e197ca-1b01-48de-a1bf-136a03b51d28` (w12-09; `docs/evidence_traces.jsonl`) |
| 5 | **trace-gap at a third-party hop — NEW, created by integration** — claims-system cannot echo the trace id | **2/38 (5.3%)**: w12-20, w12-21 (every call to that server) | Medium: cost/latency still in our trace; server side uncorrelated | `036a69d6-cc19-450c-b064-85dbabf7c23d` (w12-20; `docs/evidence_traces.jsonl`) |
| 6 | **decision-language** — output states or implies a coverage decision | **0/38 observed** on the default path (cannot occur: packet is code-composed; guard unit-tested). **Not measured on the LLM-narrative path** (no key) | would be Critical | — |

Notes
* Modes 3 and 5 are the "created by integration" modes: neither belongs to any single part (intake, tools, MCP server, logging all passed their own tests).
* Mode 1's 73% is the honest harm number. The stricter "any citation tagged with a different version" is 30/30 in state A, but that over-counts:
  state A never resolves a version at all, and most non-flood clauses read identically in both sets.
* Mode 4 is partly circular: the markers were planted by the author. 3 held-out markers is a tiny n; treat 2/3 as an existence proof, not a rate.
* After the fixes, state C (`eval`) is 0 on modes 1, 2, 3, 4 and 5-as-flagged. Modes 2 and 3 were fixed **on the same cases that found them**:
  that proves the fix, not that the next unseen case passes (see `weaknesses.md`).
