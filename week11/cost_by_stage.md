# Cost per query by stage (Week 11)

Prices: `obs.PRICES` — openai/gpt-oss-120b on Groq, $0.15/M input, $0.75/M output tokens (published list price; assumption, change in `obs.py`).
Retrieval = local compute (policy clause match / embeddings), tools = local MCP stdio subprocess calls → **$0 marginal, recorded as 0.0 on purpose**, so the split is explicit: **generation is ~100% of dollars**; tools are the time-cheapest stage, generation the slowest.

## The found request (trace `565fda46-6e45-4731-8c25-24f77d3e6aa5`, prompt `agent-v1-2026-10-09`)

| | retrieval | generation | tools |
|---|---|---|---|
| cost $ | 0.000000 | 0.000918 | 0.000000 |
| latency ms | 0.0 | 3150.8 | 3.4 |

Total: **$0.000918** · 3462 tokens in / 531 out · 3154 ms · 3 tool calls · 4 LLM laps.

Per-span: see `trace.json`.

## Fleet (real eval-suite agent runs, n = 23 on v1, 21 on v2)

| prompt | n | mean $/query | p95 $/query | mean tokens | mean latency ms | p95 latency ms |
|---|---|---|---|---|---|---|
| v1 agent-v1-2026-10-09 | 23 | 0.000579 | 0.000932 | 2648 | 7571 | 20305 |
| v2 agent-v2-2026-10-10 | 21 | 0.000524 | 0.000796 | 2442 | 12920 | 25486 |

Mean $/query by stage (v2): retrieval 0.000000 · generation 0.000524 · tools 0.000000
Mean ms/query by stage (v2): retrieval 0 · generation 12918 · tools 2.2

**Cost of the fix (measured, not assumed):** v2 vs v1 = -10% $/query, -8% tokens, +71% mean latency. Mean latency is dominated by the generation stage and is inflated by provider 429 back-off sleeps (`llm_client.chat` retries 2^n s) during back-to-back suite runs — itself the 10x evidence: single un-throttled request = ~3 s (see the found request above). Do not read the latency delta as the cost of the prompt change; the token delta is.
