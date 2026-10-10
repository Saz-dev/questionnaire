"""
Week 11 drill tooling — slice the trace log and time the find.

    python drill_tools.py start --finder me --timer-by squadmate --round 1
    python drill_tools.py find --since yesterday --input-type coverage --says-covered --no-exclusion
    python drill_tools.py find --text "groundwater"
    python drill_tools.py find --user adj-07 | --prompt-version V | --min-cost 0.001
    python drill_tools.py show <trace_id>
    python drill_tools.py found <trace_id> --slice output
    python drill_tools.py export <trace_id> --out week11/trace.json

Slices: time, user, prompt_version, input_type, cost, output (says_covered / text).
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import obs

RUNS = Path(__file__).parent / "week11" / "drill_runs.jsonl"
SEALED = Path(__file__).parent / "week11" / ".drill_sealed.json"
SLICES = ("time", "user", "prompt_version", "input_type", "cost", "output")


def _parse_since(v: str | None):
    if not v:
        return None, None
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if v == "yesterday":
        return (today - timedelta(days=1)).isoformat(), today.isoformat()
    if v == "today":
        return today.isoformat(), None
    return v, None


def read_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    return [json.loads(l) for l in RUNS.read_text().splitlines() if l.strip()]


def start_run(finder: str, timer_by: str, round_no: int) -> dict:
    run = {"event": "start", "run_id": f"r{round_no}-{int(time.time())}", "finder": finder,
           "timed_by": timer_by, "round": round_no, "started_at": time.time()}
    RUNS.parent.mkdir(exist_ok=True)
    with RUNS.open("a") as f:
        f.write(json.dumps(run) + "\n")
    return run


def finish_run(run_id: str | None, trace_id: str, slice_used: str) -> dict:
    runs = read_runs()
    starts = [r for r in runs if r["event"] == "start" and not any(
        x["event"] == "found" and x["run_id"] == r["run_id"] for x in runs)]
    start = next((r for r in starts if r["run_id"] == run_id), None) if run_id else (starts[-1] if starts else None)
    if not start:
        raise SystemExit("no open drill run — start one first")
    elapsed = time.time() - start["started_at"]
    key = json.loads(SEALED.read_text()).get(f"round{start['round']}") if SEALED.exists() else None
    rec = {"event": "found", "run_id": start["run_id"], "round": start["round"], "finder": start["finder"],
           "timed_by": start["timed_by"], "trace_id": trace_id, "slice": slice_used,
           "elapsed_s": round(elapsed, 1), "mmss": f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
           "correct": (key["trace_id"] == trace_id) if key else None, "under_5min": elapsed < 300}
    with RUNS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _fmt(r: dict) -> str:
    return (f"{r['ts'][:19]}  {r['trace_id'][:8]}  {r['user_id']:8s} {r['prompt_version']:22s} "
            f"{r['input_type']:12s} covered={r['says_covered']} excl={r['cites_exclusion']} "
            f"${r['cost_usd']:.5f}  {r['question'][:60]!r}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("start"); p.add_argument("--finder", default="me"); p.add_argument("--timer-by", default="squadmate"); p.add_argument("--round", type=int, default=1)
    p = sub.add_parser("find")
    p.add_argument("--since"); p.add_argument("--user"); p.add_argument("--prompt-version"); p.add_argument("--input-type")
    p.add_argument("--says-covered", action="store_true"); p.add_argument("--no-exclusion", action="store_true")
    p.add_argument("--text"); p.add_argument("--min-cost", type=float); p.add_argument("--limit", type=int, default=30)
    p = sub.add_parser("show"); p.add_argument("trace_id")
    p = sub.add_parser("found"); p.add_argument("trace_id"); p.add_argument("--slice", choices=SLICES, required=True); p.add_argument("--run-id")
    p = sub.add_parser("export"); p.add_argument("trace_id"); p.add_argument("--out", default="week11/trace.json")
    a = ap.parse_args()

    if a.cmd == "start":
        print(json.dumps(start_run(a.finder, a.timer_by, a.round))); print("clock running...")
    elif a.cmd == "find":
        since, until = _parse_since(a.since)
        rows = obs.search_traces(since=since, until=until, user_id=a.user, prompt_version=a.prompt_version,
                                 input_type=a.input_type, says_covered=True if a.says_covered else None,
                                 cites_exclusion=False if a.no_exclusion else None, text=a.text,
                                 min_cost=a.min_cost, limit=a.limit)
        print(f"{len(rows)} trace(s)")
        for r in rows:
            print(_fmt(r))
    elif a.cmd in ("show", "export"):
        full = obs.get_trace(a.trace_id) or next((obs.get_trace(r["trace_id"]) for r in obs.search_traces(limit=10000) if r["trace_id"].startswith(a.trace_id)), None)
        if not full:
            sys.exit("trace not found")
        if a.cmd == "show":
            print(json.dumps(full, indent=2))
        else:
            Path(a.out).write_text(json.dumps(full, indent=2)); print(f"wrote {a.out}")
    elif a.cmd == "found":
        print(json.dumps(finish_run(a.run_id, a.trace_id, a.slice), indent=2))


if __name__ == "__main__":
    main()
