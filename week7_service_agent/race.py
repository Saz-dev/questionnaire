"""
Runs both systems over the same 10 service tickets, grades each result
against ground truth, and writes:
  - race.csv               8 numbers (4 per system)
  - budget_termination.log the stress-ticket run that hits a budget
  - race_trace.log         full per-ticket tool-call trace for both systems
"""

import csv
import sys

import agent
import workflow
from data import STRESS_TICKET, TICKETS, expected_result


def grade(result: dict, expected: dict) -> bool:
    if result.get("status") != expected["status"]:
        return False
    # Some models return payout as a numeric string ("0") rather than a
    # JSON number despite the tool schema declaring it numeric — compare
    # numerically rather than penalize a formatting quirk.
    try:
        return float(result.get("payout")) == float(expected["payout"])
    except (TypeError, ValueError):
        return False


def run_system(name: str, run_fn, trace_log_lines: list[str]) -> tuple[list[dict], list[bool]]:
    results, passed = [], []
    for ticket in TICKETS:
        expected = expected_result(ticket)
        result = run_fn(ticket["ticket_id"])
        ok = grade(result, expected)
        passed.append(ok)
        results.append(result)
        trace_log_lines.append(
            f"=== {name} :: {ticket['ticket_id']} (dependency={ticket['dependency']}) "
            f"expected={expected} got status={result.get('status')} payout={result.get('payout')} "
            f"pass={ok} ==="
        )
        trace_log_lines.extend(result["trace"])
        trace_log_lines.append("")
        print(
            f"  {name:8s} {ticket['ticket_id']}  expected={expected}  "
            f"got=({result.get('status')}, {result.get('payout')})  pass={ok}"
        )
    return results, passed


def summarize(name: str, results: list[dict], passed: list[bool]) -> dict:
    latencies = sorted(r["latency_s"] for r in results)
    n = len(latencies)
    p50 = latencies[n // 2] if n % 2 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2
    total_tokens = sum(r["total_tokens"] for r in results)
    total_cost = sum(r["cost"] for r in results)
    return {
        "system": name,
        "pass_rate": sum(passed) / len(passed),
        "p50_latency_s": round(p50, 3),
        "total_tokens": total_tokens,
        "cost_per_claim": round(total_cost / len(results), 6),
    }


def main():
    trace_lines = []

    print("Running workflow over 10 tickets...")
    wf_results, wf_passed = run_system("workflow", workflow.run_ticket, trace_lines)

    print("\nRunning agent over 10 tickets...")
    ag_results, ag_passed = run_system("agent", agent.run_ticket, trace_lines)

    with open("race_trace.log", "w") as f:
        f.write("\n".join(trace_lines))

    wf_summary = summarize("workflow", wf_results, wf_passed)
    ag_summary = summarize("agent", ag_results, ag_passed)

    with open("race.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "pass_rate", "p50_latency_s", "total_tokens", "cost_per_claim"])
        writer.writeheader()
        writer.writerow(wf_summary)
        writer.writerow(ag_summary)

    print("\n=== race.csv ===")
    print(f"{'system':10s} {'pass_rate':>10s} {'p50_latency_s':>14s} {'total_tokens':>13s} {'cost_per_claim':>15s}")
    for s in (wf_summary, ag_summary):
        print(f"{s['system']:10s} {s['pass_rate']:>10.2f} {s['p50_latency_s']:>14.3f} {s['total_tokens']:>13d} {s['cost_per_claim']:>15.6f}")

    print("\nRunning budget-termination demo on the stress ticket (max_iterations=1)...")
    budget_result = agent.run_ticket(STRESS_TICKET["ticket_id"], budgets=agent.Budgets(max_iterations=1))
    with open("budget_termination.log", "w") as f:
        f.write(f"Ticket: {STRESS_TICKET['ticket_id']} (bike_model={STRESS_TICKET['bike_model']!r} is not one of the "
                f"3 manuals on file; notes point to a second unlisted model — this ticket cannot resolve)\n")
        f.write(f"Budgets: max_iterations=1 (tightened from the race default of 6, to deterministically force and show the "
                f"termination path — no ticket can be fully resolved in a single lap, so this budget is guaranteed to fire)\n\n")
        f.write("\n".join(budget_result["trace"]))
        f.write(f"\n\nFinal returned result: {budget_result}\n")
    print(f"  budget_exceeded={budget_result['budget_exceeded']!r}  final={budget_result.get('reason')}")

    print("\nWrote race.csv, budget_termination.log, race_trace.log")


if __name__ == "__main__":
    sys.exit(main())
