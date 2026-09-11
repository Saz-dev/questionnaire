"""
The agent system: an LLM-driven tool-calling loop (ReAct-style) that
decides for itself which of the 3 tools to call, in what order, and when
it's done. Four budgets are enforced every lap so a ticket that can
never resolve (see data.STRESS_TICKET) terminates cleanly instead of
spinning.
"""

import json
import time

import openai

from llm_client import chat, cost_of
from tools import TOOL_IMPLS, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are a bike warranty triage agent. For every service ticket you must, in whatever order needed:
1. Retrieve the ticket record.
2. Read the technician notes to find what the workshop's ACTUAL practice was (fuel used, part installed, torque applied, fluid used, pressure run, etc.) — it is not the same as the reported symptom.
3. Search that bike model's manual for the ONE specification the notes describe checking, and compare actual practice to it.
4. Compute the final payout.

Call get_service_ticket before anything else. Only call search_manual once you know the bike_model and the specific spec to check from the notes. Phrase the search_manual query as a direct question naming the exact spec, e.g. "What is the recommended <spec> for this bike?" or "What is the <spec> in the specifications table?" — not the reported symptom, and not a vague topic word. Only call compute_service_cost once you have parts_cost, labor_cost, and a coverage_status decision:
- coverage_status = "covered" when actual practice matches the manual's spec (a genuine defect, not misuse).
- coverage_status = "not_covered" when actual practice deviates from the manual's spec (wear, misuse, or an unauthorized part/fluid/fuel).
If a tool returns an error or "grounded": false, you may try once more with a different query, but do not repeat the exact same call more than once.

When you have the final answer, call submit_final_answer exactly once as your last action. Do not answer in plain text.
"""

SUBMIT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_final_answer",
        "description": (
            "Submit the final decision for this ticket once the payout "
            "has been computed. Ends processing — call it exactly once, "
            "as the last action, never before compute_service_cost."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "status": {"type": "string", "enum": ["covered", "not_covered", "pending_review", "error"]},
                "payout": {"type": ["number", "null"]},
                "reason": {"type": "string"},
            },
            "required": ["ticket_id", "status", "payout", "reason"],
        },
    },
}


class Budgets:
    def __init__(self, max_iterations=6, max_tokens=20000, max_cost=0.015, max_wall_clock_s=150.0):
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.max_wall_clock_s = max_wall_clock_s


DEFAULT_BUDGETS = Budgets()


def run_ticket(ticket_id: str, budgets: Budgets = DEFAULT_BUDGETS) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Process service ticket {ticket_id}."},
    ]

    trace = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    iterations = 0
    budget_hit = None
    start = time.monotonic()
    result = None

    while True:
        iterations += 1

        if iterations > budgets.max_iterations:
            budget_hit = "max_iterations"
            trace.append(f"[BUDGET] iteration {iterations} > max_iterations={budgets.max_iterations} — stopping before another LLM call")
            break

        elapsed = time.monotonic() - start
        if elapsed > budgets.max_wall_clock_s:
            budget_hit = "wall_clock"
            trace.append(f"[BUDGET] elapsed {elapsed:.1f}s > max_wall_clock_s={budgets.max_wall_clock_s} — stopping before another LLM call")
            break

        try:
            message, usage = chat(messages, tools=TOOL_SCHEMAS + [SUBMIT_TOOL_SCHEMA])
        except openai.BadRequestError as e:
            # Groq's gpt-oss-120b occasionally hallucinates a tool name
            # that was never declared (seen: "JSON", "answer") when it
            # means to call submit_final_answer. The API rejects that
            # tool call outright, but still reports what it tried to
            # submit in error.failed_generation — recover from that
            # instead of crashing the whole run.
            trace.append(f"[RECOVERED] provider rejected a hallucinated tool call: {e}")
            failed_generation = (e.body or {}).get("failed_generation", "")
            try:
                result = json.loads(failed_generation)["arguments"]
                result["ticket_id"] = ticket_id  # the hallucinated call sometimes mangles this field too
            except (json.JSONDecodeError, KeyError, TypeError):
                result = {"ticket_id": ticket_id, "status": "error", "payout": None, "reason": f"tool_call_validation_failed: {e}"}
            trace.append(f"[FINAL] {result}")
            break
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_tokens = total_prompt_tokens + total_completion_tokens
        cost = cost_of(total_prompt_tokens, total_completion_tokens)
        trace.append(
            f"[LAP {iterations}] +{usage['prompt_tokens']}/{usage['completion_tokens']} tok "
            f"(running total {total_tokens} tok, ${cost:.5f})"
        )

        if total_tokens > budgets.max_tokens:
            budget_hit = "max_tokens"
            trace.append(f"[BUDGET] total_tokens {total_tokens} > max_tokens={budgets.max_tokens} — stopping")
            break
        if cost > budgets.max_cost:
            budget_hit = "max_cost"
            trace.append(f"[BUDGET] cost ${cost:.5f} > max_cost=${budgets.max_cost} — stopping")
            break

        if message.tool_calls:
            submit_call = next((tc for tc in message.tool_calls if tc.function.name == "submit_final_answer"), None)
            if submit_call is not None:
                try:
                    result = json.loads(submit_call.function.arguments)
                except json.JSONDecodeError:
                    result = {"ticket_id": ticket_id, "status": "error", "payout": None, "reason": "unparseable submit_final_answer arguments"}
                trace.append(f"[FINAL] {result}")
                break

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
            })
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                impl = TOOL_IMPLS.get(name)
                tool_result = impl(**args) if impl else {"error": f"unknown tool {name!r}"}
                trace.append(f"  tool_call {name}({args}) -> {tool_result}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_result)})
            continue
        else:
            # The model sometimes answers in plain text instead of calling
            # submit_final_answer despite the instruction not to — try to
            # recover the same JSON contract before giving up on it.
            text = (message.content or "").strip().strip("`")
            if text.startswith("json"):
                text = text[4:]
            try:
                result = json.loads(text)
                result.setdefault("ticket_id", ticket_id)
            except json.JSONDecodeError:
                result = {"ticket_id": ticket_id, "status": "error", "payout": None, "reason": f"unparseable plain-text reply: {message.content!r}"}
            trace.append(f"[FINAL] {result}")
            break

    latency_s = time.monotonic() - start
    total_tokens = total_prompt_tokens + total_completion_tokens
    cost = cost_of(total_prompt_tokens, total_completion_tokens)

    if budget_hit is not None:
        result = {
            "ticket_id": ticket_id,
            "status": "error",
            "payout": None,
            "reason": f"terminated cleanly: {budget_hit} budget exceeded after {iterations} iteration(s)",
        }

    return {
        **result,
        "system": "agent",
        "iterations": iterations,
        "total_tokens": total_tokens,
        "cost": cost,
        "latency_s": latency_s,
        "budget_exceeded": budget_hit,
        "trace": trace,
    }


if __name__ == "__main__":
    r = run_ticket("TCK-01")
    for line in r["trace"]:
        print(line)
    print(r)
