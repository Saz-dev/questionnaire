# Week 9 — MCP agent + claims-system bolt-on (Task Set D)

MCP (Model Context Protocol) demo: an agent that **discovers** tools from
MCP servers at run time. Server two (the third-party claims system) was
bolted on with a **one-block config change** and zero agent-module edits.

```
week9_mcp/
  agent.py                 # AGENT MODULE (host): spawns servers from mcp_config.json,
                           #   tools/list discovery, LLM tool loop, tracing. UNCHANGED
                           #   between commits b04bd18..9e1f254 (see agent_diff.txt).
  mcp_client.py            # AGENT MODULE: stdio JSON-RPC client + raw wire capture.
  llm_client.py            # AGENT MODULE: OpenAI-compatible (Groq) model helper.
  run.py                   # AGENT MODULE: runner alias for agent.py.
  mcp_server.py            # tiny MCP server lib used by both servers (+gateway).
  servers/policy_server.py # SERVER ONE (ours): policy-document search; req-5 rewrite target.
  claims_thirdparty/       # SERVER TWO (third-party artifact): claims DB + claims MCP server.
  gateway.py               # BONUS: single-door gateway, audit line, scoped token.
  mcp_config.json          # CONFIG: the server registry. THIS is what changed to add server two.
  gateway_config.json      # BONUS config: one server (gateway) with --token ro.
  traces/                  # every run (tool names visible per lap) + raw wire per server.
  wire/                    # raw JSON-RPC frames captured verbatim by the transport.
  discovery_{before,after}.json  # tools/list snapshots (source of tool_counts.md).
  agent_diff.txt / config_diff.txt / wire.json / tool_counts.md /
  error_before_after.md / risk_note.md   # Task-Set-D deliverables.
```

## Run it

```bash
cd questionnaire/week9_mcp
../venv/bin/python run.py --query "What is the status of claim CLM-2024-88120 and what do the adjuster's notes say?" --name my_run
```

Takes `AI_API_KEY` / `AI_MODEL` from `questionnaire/.env` (Groq by default).
Every run writes `traces/<name>.json` (with `tools_discovered` and per-lap
`tool_calls` that name the server + tool) and raw wire frames to `wire/`.

## Deliverable pointers

| Requirement | Artifact |
|---|---|
| 1. Query that provably calls a claims tool | `traces/demo_claims.json` (calls `claim_status` + `claim_adjuster_notes`) |
| 2. Zero changed lines in the agent module | `agent_diff.txt` (b04bd18..9e1f254), `config_diff.txt` |
| 3. Tool count before -> after with names | `tool_counts.md` (sourced from `discovery_*.json` = tools/list) |
| 4. Raw initialize/tools-list/tools-call annotated + model-call location | `wire.json` |
| 5. Docstring-as-prompt + recoverable error, before/after transcript | `error_before_after.md` |
| 6. Five-line supply-chain risk note | `risk_note.md` |
| Bonus: gateway, audit line, scoped-token denial | `gateway.py`, `gateway_audit.log`, `traces/bonus_demo.json` |

## Where the model runs

Only in the host. `agent.py` calls the LLM with the discovered tool
schemas; every MCP server is pure JSON-RPC capability execution. The
`_where_the_model_call_happens` field in `wire.json` is the canonical line.