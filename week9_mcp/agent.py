"""
The Week-9 MCP agent (host).

This is the ONE file that Task Set D requires to stay untouched while
servers come and go. It is pure host: it reads the server registry in
mcp_config.json, connects to EVERY server it finds there over stdio MCP,
discovers tools via tools/list (never hard-codes a tool name), converts
them to the model's function schema, and run an LLM tool loop where each
tools/call goes back to the server that advertised the tool.

Nothing here mentions claims, policies, or any specific server. Adding a
second server is a config edit only — proven by agent_diff.txt, which is
produced by `git diff commitA commitB -- agent.py mcp_client.py
llm_client.py run.py`.

Where the model call happens: MODEL_CALL_LOCATION is the canonical line —
every lap, the host calls the LLM (llm_client.chat) with the discovered
tool schemas; the MCP servers themselves are pure JSON-RPC capability
execution, so initialize/tools/list/tools/call never touch a model.
"""

import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obs  # noqa: E402  (questionnaire/obs.py — Week 11 trace schema v2)
from llm_client import AI_MODEL, chat  # noqa: E402
from mcp_client import MCPConnection  # noqa: E402

MODEL_CALL_LOCATION = (
    "The model call happens ONLY in the host — agent.py's LLM loop issues "
    "chat.completions.create against the provider and never appears on the "
    "MCP wire: initialize, tools/list and tools/call are plain JSON-RPC between "
    "host and server, and each server's tools/call handler runs with no LLM in "
    "the path."
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "mcp_config.json")
DEFAULT_WIRE_DIR = os.path.join(HERE, "wire")

MAX_LAPS = 6

SYSTEM_PROMPT_V1 = """You are a claims triage assistant. The tools you can call were discovered
at run time from MCP servers — use exactly the tool that fits the question, and only
when the question needs it.

Rules:
- Call the tool that answers the question. Read the tool result carefully and answer
  from it — never answer from memory or guess a number.
- A tool may return an error. Errors are recoverable: they often name the correct
  input format, so correct your input once and retry. If a retry still fails, say so
  plainly and do not invent a value.
- Do not call a tool "for practice" or because it exists. Only call what the question
  requires."""

# v2 (Week 11 fix): a coverage answer must check THAT policy's own cover/exclusions
# before saying "covered". Failure it fixes: claim CLM-2024-88123 (renter policy that
# excludes flooding) was told "flood loss is covered" because the generic Endorsement W
# clause from search_policy_documents was treated as if it applied to every policy.
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT_V1 + """

Coverage rule (applies ONLY when the question asks whether a specific claim or policy
is covered — NOT to general questions about what a clause says, such as a limit or
deductible, which you answer directly from search_policy_documents):
- Identify the specific policy first: from a claim number use claim_status to get its
  policy_number; otherwise use the policy number in the question.
- Call get_policy for that policy and read its "cover" field BEFORE answering. A
  policy-specific exclusion or missing endorsement ("excludes flooding", "no water
  endorsement", "fire only") OVERRIDES generic clause text from search_policy_documents:
  a generic clause describes what an endorsement offers, not that this policy has it.
- If the policy excludes the peril or lacks the endorsement, answer "not covered" and
  name the exclusion. Only say "covered" when that policy's own cover includes it."""

# Prompt registry. Select with the AGENT_PROMPT_VERSION env var, so a canary
# or a rollback is a config flip with no deploy (see week11/prompt_version_notes.md).
PROMPTS = {
    "agent-v1-2026-10-09": SYSTEM_PROMPT_V1,
    "agent-v2-2026-10-10": SYSTEM_PROMPT_V2,
}
DEFAULT_PROMPT_VERSION = "agent-v2-2026-10-10"


def get_prompt_version() -> str:
    version = os.environ.get("AGENT_PROMPT_VERSION", DEFAULT_PROMPT_VERSION)
    if version not in PROMPTS:
        raise ValueError(f"unknown AGENT_PROMPT_VERSION {version!r}; known: {sorted(PROMPTS)}")
    return version


def _context_ids(tool: str, args: dict, result_text: str) -> list[str]:
    """Stable ids for what the model was shown: policy clauses, policy records, claim records."""
    ids: list[str] = []
    try:
        payload = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return ids
    if not isinstance(payload, dict) or "error" in payload:
        return ids
    if tool == "search_policy_documents":
        ids += [f"clause:{m['clause']}" for m in payload.get("matches", [])]
    elif tool == "get_policy":
        ids.append(f"policy:{payload.get('policy_number')}")
    elif tool in ("claim_status", "claim_adjuster_notes"):
        ids.append(f"claim:{payload.get('claim_number')}")
    return ids


def _to_openai_schema(tool: dict) -> dict:
    """MCP tool -> OpenAI function schema. The server's docstring becomes
    the description (the 'docstring-as-prompt' the model actually reads)."""
    params = tool.get("inputSchema") or {}
    if params.get("type") != "object":
        params = {"type": "object", "properties": {}, **params}
    params.setdefault("properties", {})
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": params,
        },
    }


def _wire_file(run_id: str, server_name: str) -> str:
    os.makedirs(DEFAULT_WIRE_DIR, exist_ok=True)
    return os.path.join(DEFAULT_WIRE_DIR, f"{run_id}__{server_name}.jsonl")


def discover_and_run(query: str, run_id: str | None = None, max_laps: int = MAX_LAPS,
                     user_id: str | None = None, session_id: str | None = None,
                     write_obs_trace: bool = True) -> dict:
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    prompt_version = get_prompt_version()
    rec = obs.TraceRecorder()
    context_ids: list[str] = []
    tools_called: list[str] = []

    with open(DEFAULT_CONFIG) as fh:
        config = json.load(fh)
    servers = config.get("mcpServers", {})

    connections: dict[str, MCPConnection] = {}
    discovery: dict[str, dict] = {}
    tool_to_server: dict[str, str] = {}
    try:
        # ---- connect + discover (tools/list) ----------------------------
        for sname, sconf in servers.items():
            conn = MCPConnection(sname, sconf, wire_file=_wire_file(run_id, sname), cwd=HERE)
            conn.initialize()
            tools = conn.list_tools()
            connections[sname] = conn
            discovery[sname] = {"raw_tools_list": tools}
            for t in tools:
                tool_to_server[t["name"]] = sname

        # ---- snapshot discovery for the frontend / tool-count report -----
        os.makedirs(os.path.join(HERE, "runtime"), exist_ok=True)
        with open(os.path.join(HERE, "runtime", "discovery_latest.json"), "w") as fh:
            json.dump(discovery, fh, indent=2)

        openai_tools = [t for server in discovery.values() for t in server["raw_tools_list"]]
        openai_tools = [_to_openai_schema(t) for t in openai_tools]

        messages = [
            {"role": "system", "content": PROMPTS[prompt_version]},
            {"role": "user", "content": query},
        ]

        laps: list[dict] = []
        final_answer = None
        tool_call_ids = 0

        for lap in range(1, max_laps + 1):
            t_llm = time.perf_counter()
            message, usage = chat(messages, tools=openai_tools or None)
            rec.add_span(
                f"generation.lap{lap}", "generation", (time.perf_counter() - t_llm) * 1000,
                tokens_in=usage["prompt_tokens"], tokens_out=usage["completion_tokens"],
                cost_usd=obs.llm_cost(AI_MODEL, usage["prompt_tokens"], usage["completion_tokens"]),
                lap=lap,
            )
            lap_rec = {
                "lap": lap,
                "assistant": {
                    "content": message.content,
                    "tool_calls": [
                        {
                            "name": tc.function.name,
                            "server": tool_to_server.get(tc.function.name),
                            "arguments_raw": tc.function.arguments,
                        }
                        for tc in (message.tool_calls or [])
                    ],
                },
                "model": AI_MODEL,
                "usage": usage,
            }

            if not message.tool_calls:
                final_answer = message.content
                laps.append(lap_rec)
                break

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
            })

            results = []
            for tc in message.tool_calls:
                tool_call_ids += 1
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_unparseable": tc.function.arguments}
                t_tool = time.perf_counter()
                call_result = connections[tool_to_server[name]].call_tool(name, args)
                text = "".join(c["text"] for c in call_result.get("content", []))
                rec.add_span(
                    f"tool.{name}", "tools", (time.perf_counter() - t_tool) * 1000,
                    cost_usd=obs.TOOL_CALL_COST_USD, server=tool_to_server[name],
                    is_error=bool(call_result.get("isError", False)),
                )
                tools_called.append(name)
                context_ids.extend(_context_ids(name, args, text))
                results.append({
                    "tool": name,
                    "server": tool_to_server[name],
                    "arguments": args,
                    "isError": call_result.get("isError", False),
                    "result": text,
                })
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})

            lap_rec["tool_results"] = results
            laps.append(lap_rec)

        trace = {
            "run_id": run_id,
            "query": query,
            "servers_connected": list(connections.keys()),
            "tools_discovered": {s: [t["name"] for t in d["raw_tools_list"]] for s, d in discovery.items()},
            "model": AI_MODEL,
            "model_call_location": MODEL_CALL_LOCATION,
            "laps": laps,
            "final_answer": final_answer,
            "laps_used": len(laps),
            "prompt_version": prompt_version,
        }

        if write_obs_trace:
            v2 = obs.build_trace(
                question=query, answer=final_answer or "", prompt_version=prompt_version,
                model=AI_MODEL, spans=rec.spans, retrieved_context_ids=list(dict.fromkeys(context_ids)),
                source="agent", user_id=user_id, session_id=session_id, tools_called=tools_called,
                extra={"run_id": run_id},
            )
            obs.write_trace_v2(v2)
            trace["obs_trace_id"] = v2["trace_id"]

        os.makedirs(os.path.join(HERE, "traces"), exist_ok=True)
        trace_path = os.path.join(HERE, "traces", f"{run_id}.json")
        with open(trace_path, "w") as fh:
            json.dump(trace, fh, indent=2)

        return trace
    finally:
        for conn in connections.values():
            conn.close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    query = None
    run_id = None
    if "--query" in argv:
        query = argv[argv.index("--query") + 1]
    if "--name" in argv:
        run_id = argv[argv.index("--name") + 1]
    user_id = argv[argv.index("--user") + 1] if "--user" in argv else None
    if "--help" in argv or query is None:
        print(f"usage: {sys.argv[0]} --query '...' [--name run_id]")
        return 2

    trace = discover_and_run(query, run_id=run_id or uuid.uuid4().hex[:8], user_id=user_id)
    print(json.dumps({
        "run_id": trace["run_id"],
        "query": trace["query"],
        "tools_discovered": trace["tools_discovered"],
        "laps": trace["laps_used"],
        "final_answer": trace["final_answer"],
        "prompt_version": trace["prompt_version"],
        "trace_id": trace.get("obs_trace_id"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())