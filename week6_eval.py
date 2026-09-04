"""
Week 6 — the one command that runs the whole eval set and prints pass rate by mode.

    python week6_eval.py

Runs the 4 deterministic assertions (assertions.py) over every case in
eval_set.jsonl, prints pass rate broken down by failure_mode_tag (never one
overall number - see W6-Task-Set-D.md's "common mistakes" #5), flags the 2
regression cases explicitly, and - if judge_results_v1.json / _v2.json /
labels_25.json exist on disk - also prints the judge-agreement summary so the
whole Week 6 deliverable is visible from one run.
"""
import json
from collections import defaultdict
from pathlib import Path

from assertions import run_all

EVAL_SET_PATH = Path("eval_set.jsonl")


def load_cases() -> list[dict]:
    with EVAL_SET_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def print_assertion_table(cases: list[dict]) -> None:
    by_mode = defaultdict(lambda: {"pass": 0, "total": 0})
    per_case = []
    for case in cases:
        result = run_all(case)
        per_case.append((case, result))
        mode = case["failure_mode_tag"]
        by_mode[mode]["total"] += 1
        by_mode[mode]["pass"] += result["passed"]

    print(f"\n=== Deterministic assertion pass rate by mode ({len(cases)} cases) ===")
    print(f"{'mode':30s} {'pass':>6s} {'total':>6s} {'rate':>7s}")
    total_pass = total_n = 0
    for mode, c in sorted(by_mode.items()):
        rate = c["pass"] / c["total"]
        total_pass += c["pass"]
        total_n += c["total"]
        print(f"{mode:30s} {c['pass']:6d} {c['total']:6d} {rate:6.1%}")
    print(f"{'OVERALL':30s} {total_pass:6d} {total_n:6d} {total_pass/total_n:6.1%}")

    print("\n=== Regression cases (verbatim from real failed traces) ===")
    for case, result in per_case:
        if case["regression_case"]:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {case['id']} [{status}] trace_id={case['trace_id']} - {case['question']}")


def print_judge_agreement_if_available() -> None:
    labels_path = Path("labels_25.json")
    v1_path = Path("judge_results_v1.json")
    v2_path = Path("judge_results_v2.json")
    if not labels_path.exists():
        print("\n(no labels_25.json yet - hand-label before running the judge)")
        return

    labels = {l["id"]: l["verdict"] for l in json.loads(labels_path.read_text())["labels"]}

    def agreement(results_path: Path) -> tuple[float, int, int] | None:
        if not results_path.exists():
            return None
        results = {r["id"]: r["verdict"] for r in json.loads(results_path.read_text())}
        matched = sum(1 for i, v in labels.items() if i in results and results[i] == v)
        total = sum(1 for i in labels if i in results)
        return (matched / total if total else 0.0, matched, total)

    before = agreement(v1_path)
    after = agreement(v2_path)
    print("\n=== Judge agreement with hand labels ===")
    if before:
        print(f"  agreement_before (judge_v1): {before[0]:.1%}  ({before[1]}/{before[2]})")
    else:
        print("  agreement_before: judge_results_v1.json not found yet - run `python judge.py v1`")
    if after:
        print(f"  agreement_after  (judge_v2): {after[0]:.1%}  ({after[1]}/{after[2]})")
    else:
        print("  agreement_after: judge_results_v2.json not found yet - run `python judge.py v2`")


def main() -> None:
    cases = load_cases()
    print_assertion_table(cases)
    print_judge_agreement_if_available()


if __name__ == "__main__":
    main()
