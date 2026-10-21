<!-- Soft Suave · The AI Engineering League -->
# Eval report — integrated system

Command: `python capstone.py eval --compare` (before → after) · `python capstone.py eval` (after only) · raw output in `eval_run_*.txt`.
Suite: `week12/eval_cases.jsonl`, **38 cases**, built by `week12/build_cases.py`. Expected values are computed there from the raw data
files with independent code, not from the agent. 34 cases were written first; **4 more were added after the first green run specifically to
look for failures the suite had not been tuned on** (w12-35…38). 3 of those 4 failed.

## 1. History (this is the part that matters)

| Run | File | n | Result | What it showed |
|---|---|---|---|---|
| 1 | `eval_run_1_red.txt` | 34 | 30/34 (after-config) | trace gap on 3rd-party hop (w12-20/21), ISO datetime (w12-27), "bushfire" (w12-32); DD/MM case passed outcome but had a wrong trajectory |
| 2 | `eval_run_2_after_fixes.txt` | 34 | 34/34 | all four fixed |
| 3 | `eval_run_3_unseen_cases.txt` | 38 | **35/38** | +4 unseen cases: 2 peril misclassifications → wrong money, lower-case policy number |
| 4/5 | `eval_run_4_after_loop.txt`, `eval_run_5_final.txt` | 38 | **38/38** | fixed — **on the same cases that found them** |

38/38 is therefore *not* evidence that the next unseen claim passes. It is evidence that the 3 observed defects are closed and now permanently guarded.

## 2. Pass rate per mode (final, `after`)

| mode | pass | n | | mode | pass | n |
|---|---|---|---|---|---|---|
| wrong-endorsement-version | 9 | 9 | | referral-path | 3 | 3 |
| version-control (2025 controls) | 5 | 5 | | says-decision (injection) | 3 | 3 |
| clause-miss | 4 | 4 | | pii-leak | 2 | 2 |
| money-conflation | 3 | 3 | | seam-intake | 3 | 3 |
| not-in-force | 4 | 4 | | peril-misclass | 2 | 2 |
| **overall** | **38** | **38** | | | | |

## 3. The number that moved (same 38 cases, same suite; `--before` = date filter off + redact-only logging)

| metric | before | after | n |
|---|---|---|---|
| **cited WORDING differs from the wording in force on the loss date** | **22 / 30 (73%)** | **0 / 30** | 30 cases that cite anything |
| straddle slice (loss before 2025 cutover, flood wording differs) — own score, not the average | 0 / 12 | 12 / 12 | 12 |
| PII markers leaking from held-out set, cases | 2 | 0 | 38 (3 markers) |
| overall pass | 8 / 38 | 38 / 38 | 38 |

Reading it honestly: "wrong-version tag on any citation (strict)" is 30/30 → 0/30 but over-counts (the unfiltered path never resolves a version; most clauses are identical across versions), so the headline is the wording-differs row. The 12 straddle cases all got the wrong wording before.
The "before" column also changes PII logging mode, so the rows mix two changes; each row's cause is labelled in its own mode in `taxonomy_rescored.md`.

## 4. Bonus: loss-date-aware retrieval, proven; what it cost

* 5+ straddle cases red before / green after: yes — 12 straddle cases, incl. the day before (w12-02, 2024-12-31), the cutover day (w12-03), two days before (w12-22). 0/12 → 12/12.
* Filter is code: `Wordings.retrieve(as_of=…)` drops other versions before ranking; the MCP server **refuses** calls without `as_of` (unit-tested).
* **hit-rate@3 on cases that do NOT straddle a boundary (n = 18): 18/18 with the filter, 18/18 without.** The filter cost **nothing measurable here**.
  Why that is weak evidence, not a win: each product has 13 chunks and k=3, and every non-straddle case expects a clause present in the in-force version. A cost would appear
  when the answer lives only in a version outside the filter (e.g. a lapsed-then-renewed policy relying on a withdrawn clause) — no such case exists in this slice. On a 60-page wording it may well cost something.
* With the Week-4 cross-encoder applied after the filter (`--rerank`, venv with `requirements.txt`): 38/38, same bonus numbers; mean latency 229 ms, p50 8.8 ms, max 7.6 s (model load on first call).

## 5. Judge agreement — **not met, not re-measured**

* Requirement 4 asks for the validated judge re-checked on integrated outputs, ≥85%. **I could not do this**: no `AI_API_KEY` in this checkout, and the judge needs a human-labelled integrated set.
* Correction to the Week 6 record: `prediction_outcome.md` says 80% → 84%. Recomputing from the committed `labels_25.json` and `judge_results_v1/v2.json` gives **64% (16/25) → 60% (15/25)** (the same figure `week6_eval.py` prints). The 84% is stale; the gate was **never** met on the committed files.
* Integrated outputs: the narrative is code-composed on the default path, so a groundedness judge would be nearly vacuous there; the risk it would police only exists on the `--llm` narrative path, which is guarded by `packet.guard_narrative` and the `never_states_decision` assertion instead.
* To do next: set `AI_API_KEY`, generate ~25 `--llm` narratives, hand-label, run `judge.py`, report agreement.

## 6. What this suite does NOT cover (say it before being asked)

* **Not the real corpus.** Synthetic slice: 26 wording chunks, 12 policies, 21 FNOLs, 38 cases. Not 500 claim files, not 60-page wordings, not real adjuster notes. Parse quality of the exclusions schedule is untested.
* **Not the model.** No LLM call is exercised by `eval` (planner is rules; narrative off). Tool-choice accuracy therefore measures our code, not model judgement. The `LLMPlanner` fallback and invalid-choice paths are unit-tested with a fake transport only.
* **Not semantic retrieval quality.** Scoring uses BM25; the cross-encoder is exercised but its hit-rate is not separately measured.
* **Not concurrency or load.** Single process, local stdio MCP.
* **Not multi-peril or multi-claimant FNOLs**, non-English text, scanned photos/captions, or endorsements other than the flood rewrite (only that clause differs between 2023 and 2025; H-3.1 differs by one sentence).
* **Not PII beyond the planted markers** (3 held-out markers, authored by the same person who wrote the redactor).
* **Not the third-party server's behaviour beyond `claim_status`.** `claim_adjuster_notes` is never called, so what that tool would hand over is untested here.
* Cutover date 2025-01-01 is an assumption.
