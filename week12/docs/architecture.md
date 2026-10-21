<!-- Soft Suave · The AI Engineering League -->
# Architecture — one page

```
 FNOL (id or JSON)                                    python capstone.py triage F-01
        │
        ▼
 ┌──────────────────────────── capstone.py ─────────────────────────────┐
 │ mints trace_id · opens obs.TraceRecorder (Week 11 spans/cost schema)  │
 │                                                                       │
 │  intake: normalize_loss_date, canon_policy_no, classify_peril        │  ← no model
 │                                                                       │
 │  AGENT LOOP (week12/agent.py, Week-7 shape, MAX_STEPS=8)              │
 │   planner ─► next tool ─► execute ─► state ─► planner …              │
 │   RulesPlanner (default)  |  LLMPlanner (--planner llm, falls back)   │  ← model call #1 (optional)
 │                                                                       │
 │   tool registry = LOCAL  policy_lookup                                │
 │                          coverage_in_force_check                      │
 │                          refer_to_adjuster  (terminal, forced by code)│
 │                  + MCP   tools/list at runtime from mcp_config.json   │
 │                          matched by input-schema shape, not by name   │
 └───────┬───────────────────────────────────────────────┬──────────────┘
         │ tools/call + params._meta.trace_id            │ tools/call (claim_number)
         ▼                                               ▼
 ┌──────────────────────────────┐              ┌───────────────────────────┐
 │ OUR MCP server               │              │ claims-system (3rd party,  │
 │ week12/servers/wording_server│              │ vendored, unmodified)      │
 │ search_policy_wording(       │              │ cannot echo trace id →     │
 │   query, product, as_of)     │              │ flagged `trace_gap`        │
 │ date filter IN CODE, as_of   │              │ claimant/notes stripped +  │
 │ required; logs under trace_id│              │ redacted at OUR boundary   │
 │ echoes id + server_ms        │              └───────────────────────────┘
 └──────────────┬───────────────┘
                ▼
   week12/retriever.py  BM25 (Week-4 algorithm) → [optional Week-4 cross-encoder, --rerank]
   chunks tagged product × endorsement_version × clause_id; filter BEFORE ranking
                │
                ▼
 packet.py: facts + citations (clause · endorsement version · loss date · link)
            + deductible | sub-limit | policy limit (3 fields) + referral
            + guard_narrative (decision words outside quoted wording → replaced)
                │                                    ┌─ model call #2 (optional, --llm): 2-sentence summary
                ▼                                    │   goes through the guard
        triage packet ──► traces.jsonl / traces.db (obs.write_trace_v2) ─ FNOL free text stored as sha+len only
```

## Where the model calls are
Two, both **optional and off by default**: the LLM planner (`--planner llm`) and the narrative sentence (`--llm`).
The default path makes **zero model calls**. The packet (citations, numbers, referral) is composed by code in
every mode, so a model cannot introduce a coverage decision into the parts that matter.

## Where the money goes
Default path: **$0.000000/claim** (measured, n=38). Model path: ≈ **$0.000524/claim**, borrowed from Week 11's
measured agent runs (not re-measured, no API key here). Retrieval and MCP tools are local processes: $0. See `cost_10x.md`.

## What is reused vs new
Reused unchanged: `obs.py` (spans, cost, `build_trace`, `write_trace_v2`), `week9_mcp/mcp_server.py` + `mcp_client.py` (stdio JSON-RPC),
the vendored `claims_thirdparty` server, `reranker.py`, the BM25/tokenizer algorithm from `vectordb.py`.
New: everything under `week12/` and `capstone.py`. Weeks 6–11 artifacts are untouched (their pinned reports stay valid).
