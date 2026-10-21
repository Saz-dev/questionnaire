"""One-command integrated eval: `python capstone.py eval [--before] [--compare]`.

Runs every case in eval_cases.jsonl through the SAME run_triage() the CLI uses (RAG-slice retriever
+ agent loop + MCP hop + guard + logging), applies week12/assertions.py, then reports:
  * pass rate per taxonomy mode (category), with n
  * the straddle slice on its own (not the average)
  * wrong-endorsement-version citation rate
  * PII leak counts (known vs held-out markers)
  * trajectory eval: tool-choice accuracy and the outcome-vs-trajectory gap
  * bonus: hit-rate@3 with vs without the date filter on cases that do NOT straddle a boundary
No LLM is called, so every number here is reproducible and free. The LLM judge is a separate step
(see docs/eval_report.md for its status).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from week12 import agent, assertions  # noqa: E402
from week12.mcp_host import McpSession  # noqa: E402
from week12.retriever import get_wordings  # noqa: E402

HERE = Path(__file__).parent
FN = {f["fnol_id"]: f for f in json.load(open(HERE / "data" / "fnols.json"))["fnols"]}


def load_cases() -> list[dict]:
    return [json.loads(l) for l in open(HERE / "eval_cases.jsonl") if l.strip()]


def fnol_for(case: dict) -> dict:
    base = FN[case["fnol_id"][:4]]
    return {**base, **(case.get("override") or {})}


def lcs(a: list, b: list) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def run_eval(label: str, enforce_date: bool = True, pii_mode: str = "hash", verbose: bool = True) -> dict:
    agent.PII_MODE = pii_mode
    cases = load_cases()
    rows, traces = [], []
    t_all = time.perf_counter()
    with McpSession() as sess:
        for c in cases:
            t0 = time.perf_counter()
            r = agent.run_triage(fnol_for(c), session=sess, enforce_date=enforce_date, user_id="eval")
            ms = (time.perf_counter() - t0) * 1000
            a = assertions.run_all(r, c)
            exp_traj = c["expect"]["trajectory"]
            act_traj = [t["tool"] for t in r["trajectory"]]
            wrong_wording = _wrong_wording(r["packet"])
            rows.append({"id": c["id"], "wrong_wording": wrong_wording, "category": c["category"], "straddle": c["straddle"], "note": c["note"],
                         "passed": a["passed"], "failed": a["failed"], "checks": a["checks"],
                         "pii_leaks": a["pii_leaks"], "trace_id": r["trace"]["trace_id"],
                         "expected_trajectory": exp_traj, "actual_trajectory": act_traj,
                         "trajectory_exact": exp_traj == act_traj,
                         "traj_lcs": lcs(exp_traj, act_traj), "traj_expected_len": len(exp_traj),
                         "cited_versions": [x["endorsement_version"] for x in r["packet"]["citations"]],
                         "flags": r["packet"]["flags"], "latency_ms": round(ms, 1),
                         "cost_usd": r["trace"]["totals"]["cost_usd"]})
            traces.append(r["trace"])
    wall = time.perf_counter() - t_all
    out = HERE / "runtime" / f"eval_{label}.json"
    (HERE / "runtime").mkdir(exist_ok=True)
    with open(HERE / "runtime" / f"eval_traces_{label}.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(t) for t in traces) + "\n")
    s = summarize(rows, label, enforce_date, pii_mode, wall)
    out.write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
    if verbose:
        print_report(s, rows)
    return {"summary": s, "rows": rows}


def _wrong_wording(pkt: dict) -> bool | None:
    """True if any cited clause's TEXT differs from the same clause in the version in force on the loss date
    (i.e. the adjuster would read different words than the policy actually carried that day)."""
    w = get_wordings()
    ld = (pkt["loss_date"] or "")[:10]
    if not pkt["citations"] or len(ld) != 10 or ld[4] != "-":
        return None
    inforce = w.version_in_force(ld)
    by_id = {c["chunk_id"]: c["text"] for c in w.chunks}
    for c in pkt["citations"]:
        prod, _ver, clause = c["chunk_id"].split("/")
        ref = by_id.get(f"{prod}/{inforce}/{clause}")
        if ref is None or ref != c["text"]:
            return True
    return False


def rate(rows: list[dict], pred) -> tuple[int, int]:
    sel = [r for r in rows if pred(r)]
    return sum(r["passed"] for r in sel), len(sel)


def summarize(rows, label, enforce_date, pii_mode, wall) -> dict:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    version_rows = [r for r in rows if r["checks"]["cited_version_matches_loss_date"] is not None]
    wrong_ver = [r for r in version_rows if r["checks"]["cited_version_matches_loss_date"] is False]
    stra = [r for r in rows if r["straddle"]]
    pii_known = sum(1 for r in rows if r["pii_leaks"]["known"])
    pii_held = sum(1 for r in rows if r["pii_leaks"]["heldout"])
    n_held = sum(1 for r in rows if r["pii_leaks"]["heldout"] is not None and r["id"] and True)
    tool_acc = sum(r["traj_lcs"] for r in rows) / sum(r["traj_expected_len"] for r in rows)
    outcome = sum(r["passed"] for r in rows) / len(rows)
    traj = sum(r["trajectory_exact"] for r in rows) / len(rows)
    disagree = [r["id"] for r in rows if r["passed"] != r["trajectory_exact"]]
    lat = [r["latency_ms"] for r in rows]
    return {
        "label": label, "date_filter": enforce_date, "pii_mode": pii_mode, "n": len(rows),
        "pass": sum(r["passed"] for r in rows), "wall_s": round(wall, 2),
        "per_category": {k: [sum(x["passed"] for x in v), len(v)] for k, v in sorted(by_cat.items())},
        "straddle_slice": [sum(r["passed"] for r in stra), len(stra)],
        "wrong_version_citation": [len(wrong_ver), len(version_rows)],
        "wrong_wording": [sum(1 for r in rows if r["wrong_wording"]), sum(1 for r in rows if r["wrong_wording"] is not None)],
        "never_states_decision_fail": sum(1 for r in rows if r["checks"]["never_states_decision"] is False),
        "pii": {"cases_with_known_leak": pii_known, "cases_with_heldout_leak": pii_held,
                "heldout_markers_total": sum(len(c["pii_heldout"]) for c in load_cases())},
        "trajectory": {"tool_choice_accuracy": round(tool_acc, 3), "exact_trajectory_pass": round(traj, 3),
                       "outcome_pass": round(outcome, 3), "gap_outcome_minus_trajectory_pp": round((outcome - traj) * 100, 1),
                       "disagreeing_cases": disagree},
        "latency_ms": {"mean": round(statistics.mean(lat), 1), "p50": round(statistics.median(lat), 1),
                       "max": round(max(lat), 1)},
        "failed_cases": [r["id"] for r in rows if not r["passed"]],
    }


def print_report(s: dict, rows: list[dict]) -> None:
    print(f"\n=== Week 12 integrated eval [{s['label']}]  n={s['n']}  date_filter={s['date_filter']}  pii_mode={s['pii_mode']} ===")
    print(f"{'mode (category)':30s} {'pass':>5s} {'n':>4s} {'rate':>7s}")
    for k, (p, n) in s["per_category"].items():
        print(f"{k:30s} {p:5d} {n:4d} {p / n:7.1%}")
    print(f"{'OVERALL':30s} {s['pass']:5d} {s['n']:4d} {s['pass'] / s['n']:7.1%}")
    p, n = s["straddle_slice"]
    print(f"\nSTRADDLE slice (loss before 2025 cutover, flood wording differs): {p}/{n} = {p / n:.1%}")
    w, n = s["wrong_version_citation"]
    print(f"citations tagged with an endorsement version other than the one in force (strict): {w}/{n} = {w / n:.1%}")
    ww, nn = s["wrong_wording"]
    print(f"WRONG WORDING cited (cited text differs from the in-force text; the real harm): {ww}/{nn} = {ww / nn:.1%}")
    print(f"cases where output contained a decision word: {s['never_states_decision_fail']}")
    print(f"PII: cases leaking KNOWN markers={s['pii']['cases_with_known_leak']}, "
          f"cases leaking HELD-OUT markers={s['pii']['cases_with_heldout_leak']} "
          f"(held-out markers planted: {s['pii']['heldout_markers_total']})")
    t = s["trajectory"]
    print(f"TRAJECTORY: tool-choice accuracy={t['tool_choice_accuracy']:.1%}  exact-sequence pass={t['exact_trajectory_pass']:.1%}  "
          f"outcome pass={t['outcome_pass']:.1%}  gap(outcome-trajectory)={t['gap_outcome_minus_trajectory_pp']:+.1f}pp  "
          f"disagree={t['disagreeing_cases']}")
    print(f"latency per triage (incl. MCP hop): mean {s['latency_ms']['mean']} ms, p50 {s['latency_ms']['p50']} ms, max {s['latency_ms']['max']} ms")
    bad = [r for r in rows if not r["passed"]]
    if bad:
        print("\nFAILED CASES:")
        for r in bad:
            print(f"  {r['id']} [{r['category']}] {r['note']}\n      failed={r['failed']} versions={r['cited_versions']} trace={r['trace_id']}")


def bonus_hit_rate() -> dict:
    """What did the date filter cost in hit-rate@3 on cases that do NOT straddle a version boundary?"""
    w = get_wordings()
    cases = [c for c in load_cases() if c["expect"].get("clause") and c["expect"].get("version") and not c["straddle"]]
    res = {}
    for name, enforce in (("with_date_filter", True), ("without_date_filter", False)):
        strict = loose = 0
        for c in cases:
            f = fnol_for(c)
            peril = agent.classify_peril(f["text"])
            if peril is None:
                continue
            prod = "household" if f["policy_no"].startswith(("HH", "POL")) else "motor"
            hits = w.retrieve(agent.PERIL_QUERY[peril], product=prod, as_of=f["loss_date"][:10], k=3, enforce_date=enforce)["hits"]
            loose += any(h["clause_id"] == c["expect"]["clause"] for h in hits)
            strict += any(h["clause_id"] == c["expect"]["clause"] and h["version"] == c["expect"]["version"] for h in hits)
        res[name] = {"hit@3_clause_only": f"{loose}/{len(cases)}", "hit@3_clause_and_version": f"{strict}/{len(cases)}"}
    res["n_non_straddle_cases"] = len(cases)
    return res


def compare() -> None:
    before = run_eval("before", enforce_date=False, pii_mode="redact", verbose=False)
    after = run_eval("after", enforce_date=True, pii_mode="hash", verbose=False)
    b, a = before["summary"], after["summary"]
    print("\n=== BEFORE -> AFTER (same cases, same suite) ===")
    for name, key in (("wrong WORDING cited (text differs)", "wrong_wording"),
                      ("wrong-version tag on any citation (strict)", "wrong_version_citation"),
                      ("straddle slice pass", "straddle_slice")):
        print(f"{name:42s} {b[key][0]}/{b[key][1]}  ->  {a[key][0]}/{a[key][1]}")
    print(f"{'PII held-out leak (cases)':42s} {b['pii']['cases_with_heldout_leak']}  ->  {a['pii']['cases_with_heldout_leak']}")
    print(f"{'overall pass':42s} {b['pass']}/{b['n']}  ->  {a['pass']}/{a['n']}")
    print("\nBonus - hit-rate@3 on the non-straddle cases:")
    print(json.dumps(bonus_hit_rate(), indent=1))
    print_report(a, after["rows"])


if __name__ == "__main__":
    compare()
