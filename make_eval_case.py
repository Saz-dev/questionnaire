"""
Turn a found trace into a Week-6-suite eval case (failure -> test).

    python make_eval_case.py <trace_id> --truth excluded --keywords "not covered,exclud"
    python make_eval_case.py --guard "Is flood damage covered under POL-2024-88120?" --truth covered --keywords covered

The case keeps the REAL question text from the failing trace (un-redacted question
is not stored in traces, so pass --question if the redacted one lost the claim id).
Written to week11/eval_case.jsonl and appended to eval_set.jsonl.
"""
import argparse, json
from pathlib import Path
import obs

p = argparse.ArgumentParser()
p.add_argument("trace_id", nargs="?")
p.add_argument("--question"); p.add_argument("--id", required=True)
p.add_argument("--truth", choices=["excluded", "covered"], required=True)
p.add_argument("--keywords", default=""); p.add_argument("--tag", default="exclusion-ignored")
p.add_argument("--regression", action="store_true")
a = p.parse_args()

t = obs.get_trace(a.trace_id) if a.trace_id else None
case = {
    "id": a.id, "suite": "claims-agent", "live_agent": True,
    "trace_id": a.trace_id or "", "question": a.question or t["question"],
    "retrieval_mode": "agent", "failure_mode_tag": a.tag, "regression_case": a.regression,
    "judge_eligible": False, "expected_grounded": True,
    "expected_keywords": [k.strip() for k in a.keywords.split(",") if k.strip()] or None,
    "expected_sources": None, "evidence_chunk_ids": (t or {}).get("retrieved_context_ids", []),
    "coverage_truth": a.truth, "grounded": True,
    "answer": (t or {}).get("answer", ""),
}
Path("week11").mkdir(exist_ok=True)
with Path("week11/eval_case.jsonl").open("a") as f:
    f.write(json.dumps(case) + "\n")
with Path("eval_set.jsonl").open("a") as f:
    f.write(json.dumps(case) + "\n")
print("added", case["id"])
