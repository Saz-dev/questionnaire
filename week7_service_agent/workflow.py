"""
The fixed-workflow system: hard-coded step order, same three tools, same
model, same output contract as agent.py — no loop, no tool-calling, no
decision by the model about what to do next. Two judgement calls that
genuinely need language understanding are still made by the model, each
in exactly one completion call, in a fixed position, never repeated:
  (a) turning the technician notes into a manual search query
  (b) comparing actual practice against the retrieved spec
Everything else — which tool runs when, and the final cost arithmetic —
is plain Python, not a model decision.
"""

import json
import time

from llm_client import chat, cost_of
from tools import compute_service_cost, get_service_ticket, search_manual

QUERY_PROMPT = """Read the technician notes below. Identify the ONE specification \
the notes describe checking (not the reported symptom).

Reply with ONLY JSON, no other text: {"query": "<question>"}
Phrase <question> as a direct question naming the exact spec, e.g. \
"What is the recommended <spec> for this bike?" or "What is the <spec> in the \
specifications table?" — never the reported symptom, never a vague topic word.
"""

COMPARE_PROMPT = """Read the technician notes and the manual excerpts below. \
Decide whether the workshop's ACTUAL practice matches the manual's specification.

Reply with ONLY JSON, no other text: {"coverage_status": "covered" | "not_covered", "matched_value": "<the spec value found>"}
"covered" means actual practice matches the spec (a genuine defect, not misuse).
"not_covered" means actual practice deviates from the spec (wear, misuse, or an unauthorized part/fluid/fuel).
"""


def _generate_query(technician_notes: str) -> tuple[str, dict]:
    messages = [
        {"role": "system", "content": QUERY_PROMPT},
        {"role": "user", "content": f"Technician notes: {technician_notes}"},
    ]
    message, usage = chat(messages)
    text = (message.content or "").strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    try:
        query = json.loads(text)["query"]
    except (json.JSONDecodeError, KeyError, TypeError):
        query = technician_notes[:80]
    return query, usage


def _decide_coverage(technician_notes: str, excerpts: list[str]) -> tuple[str, str, dict]:
    messages = [
        {"role": "system", "content": COMPARE_PROMPT},
        {"role": "user", "content": f"Technician notes: {technician_notes}\n\nManual excerpts:\n" + "\n---\n".join(excerpts)},
    ]
    message, usage = chat(messages)
    text = (message.content or "").strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    try:
        parsed = json.loads(text)
        coverage_status = parsed["coverage_status"]
        matched_value = parsed.get("matched_value", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        coverage_status, matched_value = "pending_review", ""
    return coverage_status, matched_value, usage


def run_ticket(ticket_id: str) -> dict:
    start = time.monotonic()
    trace = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Step 1 — get_service_ticket (hard-coded: always first, always this tool).
    ticket = get_service_ticket(ticket_id)
    trace.append(f"[STEP 1] get_service_ticket({ticket_id!r}) -> {ticket}")
    if "error" in ticket:
        latency_s = time.monotonic() - start
        return {
            "ticket_id": ticket_id, "status": "error", "payout": None,
            "reason": ticket["error"], "system": "workflow", "iterations": 1,
            "total_tokens": 0, "cost": 0.0, "latency_s": latency_s,
            "budget_exceeded": None, "trace": trace,
        }

    # Step 2 — one model call to turn the notes into a manual query
    # (hard-coded: always second, always this one call, never repeated).
    query, usage1 = _generate_query(ticket["technician_notes"])
    total_prompt_tokens += usage1["prompt_tokens"]
    total_completion_tokens += usage1["completion_tokens"]
    trace.append(f"[STEP 2] generate_query -> {query!r} ({usage1})")

    # Step 3 — search_manual (hard-coded: always third, always this tool).
    manual_result = search_manual(ticket["bike_model"], query)
    trace.append(f"[STEP 3] search_manual(...) -> {manual_result}")
    if not manual_result.get("grounded"):
        latency_s = time.monotonic() - start
        total_tokens = total_prompt_tokens + total_completion_tokens
        return {
            "ticket_id": ticket_id, "status": "error", "payout": None,
            "reason": manual_result.get("message", manual_result.get("error", "no manual match")),
            "system": "workflow", "iterations": 1, "total_tokens": total_tokens,
            "cost": cost_of(total_prompt_tokens, total_completion_tokens),
            "latency_s": latency_s, "budget_exceeded": None, "trace": trace,
        }

    # Step 4 — one model call to compare notes against the retrieved spec
    # (hard-coded: always fourth, always this one call, never repeated).
    coverage_status, matched_value, usage2 = _decide_coverage(ticket["technician_notes"], manual_result["excerpts"])
    total_prompt_tokens += usage2["prompt_tokens"]
    total_completion_tokens += usage2["completion_tokens"]
    trace.append(f"[STEP 4] decide_coverage -> {coverage_status!r} (matched_value={matched_value!r}) ({usage2})")

    # Step 5 — compute_service_cost (hard-coded: always fifth, always this tool).
    payout_result = compute_service_cost(ticket["parts_cost"], ticket["labor_cost"], coverage_status)
    trace.append(f"[STEP 5] compute_service_cost(...) -> {payout_result}")

    reason = f"actual practice {'matches' if coverage_status == 'covered' else 'deviates from'} manual spec ({matched_value})"

    latency_s = time.monotonic() - start
    total_tokens = total_prompt_tokens + total_completion_tokens
    cost = cost_of(total_prompt_tokens, total_completion_tokens)

    return {
        "ticket_id": ticket_id,
        "status": coverage_status,
        "payout": payout_result["payout"],
        "reason": reason,
        "system": "workflow",
        "iterations": 1,
        "total_tokens": total_tokens,
        "cost": cost,
        "latency_s": latency_s,
        "budget_exceeded": None,
        "trace": trace,
    }


if __name__ == "__main__":
    r = run_ticket("TCK-01")
    for line in r["trace"]:
        print(line)
    print(r)
