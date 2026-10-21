#!/usr/bin/env python
"""Week 12 capstone: the ONE entry point for the Claims Triage & Coverage Assistant.

    python capstone.py triage F-01            one FNOL in -> triage packet + referral out (one trace id)
    python capstone.py triage F-01 --before   same, with the date filter OFF (reproduces the failure)
    python capstone.py eval                   the whole integrated eval suite, before -> after
    python capstone.py selftest               unit/integration tests (no network, no key)
    python capstone.py trace <trace_id>       per-span cost + latency, hops, pinned inputs
    python capstone.py replay <trace_id>      re-run a trace's inputs and prove the same packet comes out
    python capstone.py cost                   cost per claim + the 10x / hailstorm plan with numbers
    python capstone.py demo                   the 10-minute four-part demo, in order

The system TRIAGES AND CITES. It never states a coverage decision: every run ends in refer_to_adjuster.
Runs on Python 3.9+ with nothing else for everything above (stdlib only). --rerank uses the Week-4 cross-encoder
(needs `pip install -r requirements.txt`); --llm / --planner llm need AI_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from week12 import agent, evalrun  # noqa: E402
from week12.mcp_host import McpSession  # noqa: E402

DATA = ROOT / "week12" / "data"
RUNTIME = ROOT / "week12" / "runtime"
MAIN_LOG = ROOT / "traces.jsonl"


def load_fnol(ref: str) -> dict:
    p = Path(ref)
    if p.exists():
        f = json.loads(p.read_text())
        f.setdefault("fnol_id", p.stem)
        f["_ref"] = {"path": str(p.resolve())}
        return f
    for f in json.load(open(DATA / "fnols.json"))["fnols"]:
        if f["fnol_id"] == ref:
            return {**f, "_ref": {"fnol_id": ref}}
    sys.exit(f"unknown FNOL {ref!r}: pass an id from week12/data/fnols.json (F-01..F-21) or a path to a JSON file")


def cmd_triage(a) -> int:
    import obs
    fnol = load_fnol(a.fnol)
    if a.rerank:
        os.environ["W12_RERANK"] = "1"
    planner = agent.LLMPlanner() if a.planner == "llm" else agent.RulesPlanner()
    r = agent.run_triage(fnol, enforce_date=not a.before, planner=planner, narrative_llm=a.llm)
    trace = r["trace"]
    trace["fnol_ref"] = fnol["_ref"]
    obs.write_trace_v2(trace)                     # the repo's one searchable log (traces.jsonl + traces.db)
    if a.json:
        print(json.dumps({k: v for k, v in r["packet"].items() if k != "text"}, indent=1))
    else:
        print(r["packet"]["text"])
    print_spans(trace)
    return 0


def print_spans(t: dict) -> None:
    print(f"\ntrace_id={t['trace_id']}  planner={t['planner']}  data_hash={t['data_hash']}  "
          f"wording_version={t['wording_version']}  date_filter={t['enforce_date']}")
    print(f"{'span':42s} {'kind':10s} {'ms':>9s} {'$':>10s}  side")
    for s in t["spans"]:
        print(f"{s['name']:42s} {s['kind']:10s} {s['duration_ms']:9.2f} {s['cost_usd']:10.6f}  {s['attrs'].get('side', '')}")
    tot = t["totals"]
    print(f"{'TOTAL':42s} {'':10s} {tot['latency_ms']:9.2f} {tot['cost_usd']:10.6f}")
    for e in t.get("server_echo", []):
        print(f"  hop {e['server']}.{e['tool']}: server echoed trace_id == ours: {e['echo_trace_id'] == t['trace_id']}")
    if t.get("trace_gap"):
        print(f"  trace gap (server cannot echo the id; client-side span only): {t['trace_gap']}")


def find_trace(trace_id: str) -> dict | None:
    for path in [MAIN_LOG, ROOT / "week12" / "docs" / "evidence_traces.jsonl", *sorted(RUNTIME.glob("*traces*.jsonl"))]:
        if path.exists():
            for line in path.read_text().splitlines():
                if trace_id in line:
                    t = json.loads(line)
                    if t["trace_id"] == trace_id:
                        return t
    return None


def cmd_trace(a) -> int:
    t = find_trace(a.trace_id)
    if not t:
        sys.exit(f"trace {a.trace_id} not found in traces.jsonl, week12/docs/evidence_traces.jsonl or week12/runtime/*traces*.jsonl")
    print_spans(t)
    print("\nPinned for reproducibility:", json.dumps({k: t.get(k) for k in (
        "fnol_id", "planner", "model", "prompt_version", "wording_version", "data_hash", "packet_sha",
        "retrieved_context_ids", "enforce_date", "flags")}, indent=1))
    print("question (as logged):", t["question"])
    return 0


def cmd_replay(a) -> int:
    t = find_trace(a.trace_id)
    if not t:
        sys.exit("trace not found")
    if t.get("data_hash") != agent.data_hash():
        sys.exit(f"data changed since the trace (trace {t.get('data_hash')} vs now {agent.data_hash()}): "
                 "wording/policy files differ, so the original answer cannot be reproduced byte-for-byte")
    ref = t.get("fnol_ref")
    if ref and "fnol_id" in ref:
        fnol = load_fnol(ref["fnol_id"])
    elif ref and "path" in ref:
        fnol = load_fnol(ref["path"])
    else:   # eval traces: rebuild the input from the case file by fnol_id
        row = next((r for f in RUNTIME.glob("eval_*.json") for r in json.loads(f.read_text()).get("rows", [])
                    if r["trace_id"] == a.trace_id), None)
        if not row:
            sys.exit("cannot resolve the input for this trace")
        fnol = evalrun.fnol_for(next(c for c in evalrun.load_cases() if c["id"] == row["id"]))
    r = agent.run_triage(fnol, enforce_date=t["enforce_date"])
    same = r["packet"]["sha"] == t["packet_sha"]
    print(f"original packet_sha={t['packet_sha']}  replay packet_sha={r['packet']['sha']}  identical={same}")
    print(f"same wording version: {r['trace']['wording_version'] == t['wording_version']}  "
          f"same retrieved chunks: {r['trace']['retrieved_context_ids'] == t['retrieved_context_ids']}")
    return 0 if same else 1


def cmd_eval(a) -> int:
    if a.rerank:
        os.environ["W12_RERANK"] = "1"
    if a.before:
        evalrun.run_eval("before", enforce_date=False, pii_mode="redact")
    elif a.compare:
        evalrun.compare()
    else:
        evalrun.run_eval("after", enforce_date=True, pii_mode="hash")
        print("\nbonus hit-rate@3 (non-straddle cases):", json.dumps(evalrun.bonus_hit_rate()))
    return 0


def cmd_selftest(a) -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "week12" / "tests"), top_level_dir=str(ROOT))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


def cmd_cost(a) -> int:
    from week12 import cost
    cost.report()
    return 0


def cmd_demo(a) -> int:
    from week12 import demo
    demo.run()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage"); t.add_argument("fnol"); t.add_argument("--before", action="store_true")
    t.add_argument("--llm", action="store_true"); t.add_argument("--planner", choices=("rules", "llm"), default="rules")
    t.add_argument("--rerank", action="store_true"); t.add_argument("--json", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("--before", action="store_true"); e.add_argument("--compare", action="store_true")
    e.add_argument("--rerank", action="store_true")
    sub.add_parser("selftest")
    tr = sub.add_parser("trace"); tr.add_argument("trace_id")
    rp = sub.add_parser("replay"); rp.add_argument("trace_id")
    sub.add_parser("cost"); sub.add_parser("demo")
    a = ap.parse_args()
    return {"triage": cmd_triage, "eval": cmd_eval, "selftest": cmd_selftest, "trace": cmd_trace,
            "replay": cmd_replay, "cost": cmd_cost, "demo": cmd_demo}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
