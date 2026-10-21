<!-- Soft Suave · The AI Engineering League -->
# Trajectory eval (Week 8 method) on the integrated agent

Method: for each of the 38 cases `eval_cases.jsonl` stores the *expected tool sequence* (computed independently in `build_cases.py`, e.g.
`policy_lookup → coverage_in_force_check → search_policy_wording → refer_to_adjuster`; lapsed policy: no wording search; unknown policy / missing or ambiguous date: `policy_lookup → refer_to_adjuster`).
* **tool-choice accuracy** = Σ expected tools matched in order (LCS) / Σ expected tools.
* **trajectory pass** = actual sequence exactly equals expected.
* **outcome pass** = every outcome assertion in `assertions.py` passes (decision-free, cites clause+version, version matches loss date, in-force right, money right, no PII, trace survives).
* **gap** = outcome pass − trajectory pass, in percentage points.

## Numbers (n = 38 unless noted)

| run | tool-choice acc | exact trajectory | outcome pass | gap (outcome − trajectory) | cases where they disagree |
|---|---|---|---|---|---|
| first build, 34 cases (`eval_run_1_red.txt`) | 98.4% | 91.2% | 88.2% | **−2.9 pp** | w12-20, w12-21, w12-26 |
| + 4 unseen cases (`eval_run_3_unseen_cases.txt`) | 98.6% | 97.4% | 92.1% | **−5.3 pp** | w12-35, w12-36 |
| final (`eval_run_5_final.txt`) | 100% | 100% | 100% | 0.0 pp | — |
| date filter off (state A) | 100% | 100% | 21.1% | **−78.9 pp** | 30 cases |

## The gap is the finding
* **Trajectory looked perfect while outcomes were wrong** (−5.3 pp, w12-35/36): the agent called exactly the right tools in the right order and still handed the adjuster the **wrong peril's deductible and sub-limit**. Tool-choice accuracy cannot see a bad classification inside a correct tool sequence.
* **Outcome looked fine while the trajectory was wrong** (w12-26, `DD/MM/YYYY`): every outcome assertion passed, but the agent had called the in-force tool with a date it should never have guessed. Outcome checks alone would have shipped it.
* **State A (date filter off): tool-choice accuracy 100%, outcome 21%.** The agent's *process* was identical; only the retriever's date handling differed. A trajectory-only eval would have called that system perfect.

Caveat: planner is rules-based, so 100% tool-choice accuracy is a statement about our code. With `--planner llm` this eval is the one to run; not exercised here (no key).
