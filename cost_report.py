"""
Week 11 — cost per query by stage + the 10x question, from REAL spans in traces.db.

    python cost_report.py [--trace-id ID] [--today-per-min 1.0] [--write]

Stages: retrieval / generation / tools. Attribution first, optimisation second.
"""
import argparse, json, statistics as st
import obs

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0

def stage_costs(t):
    return {k: t["totals"]["cost_by_stage"].get(k, 0.0) for k in ("retrieval", "generation", "tools")}

def stage_latency(t):
    out = {"retrieval": 0.0, "generation": 0.0, "tools": 0.0}
    for s in t["spans"]:
        out[s["kind"]] = out.get(s["kind"], 0.0) + s["duration_ms"]
    return out

def fleet(source="agent", prompt_version=None, user_prefix=None):
    rows = obs.search_traces(source=source, prompt_version=prompt_version, limit=100000)
    ts = [obs.get_trace(r["trace_id"]) for r in rows]
    return [t for t in ts if t["spans"] and (not user_prefix or str(t["user_id"]).startswith(user_prefix))]

def summarize(ts):
    cost = [t["totals"]["cost_usd"] for t in ts]
    lat = [t["totals"]["latency_ms"] for t in ts]
    tok = [t["totals"]["tokens_in"] + t["totals"]["tokens_out"] for t in ts]
    stages = {k: st.mean(stage_costs(t)[k] for t in ts) for k in ("retrieval", "generation", "tools")}
    lats = {k: st.mean(stage_latency(t)[k] for t in ts) for k in ("retrieval", "generation", "tools")}
    return {"n": len(ts), "cost_mean": st.mean(cost), "cost_p95": pct(cost, .95), "lat_mean": st.mean(lat),
            "lat_p95": pct(lat, .95), "tok_mean": st.mean(tok), "tok_p95": pct(tok, .95),
            "stage_cost": stages, "stage_lat": lats}

def tenx(s, today_per_min):
    f = 10
    rpm = today_per_min * f
    tpm = rpm * s["tok_mean"]
    return {"rpm": rpm, "tpm": tpm, "tpm_limit": obs.RATE_LIMIT_TPM, "tpm_ratio": tpm / obs.RATE_LIMIT_TPM,
            "rpm_ratio": rpm / obs.RATE_LIMIT_RPM,
            "usd_day": rpm * 60 * 24 * s["cost_mean"],
            "concurrency": rpm / 60 * (s["lat_mean"] / 1000),   # Little's law: busy workers
            "max_rpm_by_tpm": obs.RATE_LIMIT_TPM / s["tok_mean"]}

def md_row(name, d, unit="$", fmt="{:.6f}"):
    return f"| {name} | " + " | ".join(fmt.format(d[k]) for k in ("retrieval", "generation", "tools")) + " |"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-id", default="565fda46-6e45-4731-8c25-24f77d3e6aa5")
    ap.add_argument("--today-per-min", type=float, default=1.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    t = obs.get_trace(a.trace_id)
    one = stage_costs(t); onel = stage_latency(t)
    v1 = fleet("agent", "agent-v1-2026-10-09", "eval-"); v2 = fleet("agent", "agent-v2-2026-10-10", "eval-")
    s1, s2 = summarize(v1), summarize(v2)
    x = tenx(s2, a.today_per_min)

    md = f"""# Cost per query by stage (Week 11)

Prices: `obs.PRICES` — openai/gpt-oss-120b on Groq, $0.15/M input, $0.75/M output tokens (published list price; assumption, change in `obs.py`).
Retrieval = local compute (policy clause match / embeddings), tools = local MCP stdio subprocess calls → **$0 marginal, recorded as 0.0 on purpose**, so the split is explicit: **generation is ~100% of dollars**; tools are the time-cheapest stage, generation the slowest.

## The found request (trace `{a.trace_id}`, prompt `{t['prompt_version']}`)

| | retrieval | generation | tools |
|---|---|---|---|
{md_row('cost $', one)}
{md_row('latency ms', onel, fmt='{:.1f}')}

Total: **${t['totals']['cost_usd']:.6f}** · {t['totals']['tokens_in']} tokens in / {t['totals']['tokens_out']} out · {t['totals']['latency_ms']:.0f} ms · {len(t['tools_called'])} tool calls · {len([s for s in t['spans'] if s['kind']=='generation'])} LLM laps.

Per-span: see `trace.json`.

## Fleet (real eval-suite agent runs, n = {s1['n']} on v1, {s2['n']} on v2)

| prompt | n | mean $/query | p95 $/query | mean tokens | mean latency ms | p95 latency ms |
|---|---|---|---|---|---|---|
| v1 agent-v1-2026-10-09 | {s1['n']} | {s1['cost_mean']:.6f} | {s1['cost_p95']:.6f} | {s1['tok_mean']:.0f} | {s1['lat_mean']:.0f} | {s1['lat_p95']:.0f} |
| v2 agent-v2-2026-10-10 | {s2['n']} | {s2['cost_mean']:.6f} | {s2['cost_p95']:.6f} | {s2['tok_mean']:.0f} | {s2['lat_mean']:.0f} | {s2['lat_p95']:.0f} |

Mean $/query by stage (v2): retrieval {s2['stage_cost']['retrieval']:.6f} · generation {s2['stage_cost']['generation']:.6f} · tools {s2['stage_cost']['tools']:.6f}
Mean ms/query by stage (v2): retrieval {s2['stage_lat']['retrieval']:.0f} · generation {s2['stage_lat']['generation']:.0f} · tools {s2['stage_lat']['tools']:.1f}

**Cost of the fix (measured, not assumed):** v2 vs v1 = {s2['cost_mean']/s1['cost_mean']*100-100:+.0f}% $/query, {s2['tok_mean']/s1['tok_mean']*100-100:+.0f}% tokens, {s2['lat_mean']/s1['lat_mean']*100-100:+.0f}% mean latency. Mean latency is dominated by the generation stage and is inflated by provider 429 back-off sleeps (`llm_client.chat` retries 2^n s) during back-to-back suite runs — itself the 10x evidence: single un-throttled request = ~3 s (see the found request above). Do not read the latency delta as the cost of the prompt change; the token delta is.
"""
    ten = f"""# What breaks first at 10x claim volume

**Rate limit (tokens/minute) breaks first — not cost, not latency.** At {x['max_rpm_by_tpm']:.1f} requests/min the provider's {obs.RATE_LIMIT_TPM:,} TPM cap is already hit (mean {s2['tok_mean']:.0f} tokens/query); assuming today is {a.today_per_min:g} claim/min, 10x = {x['rpm']:g}/min = {x['tpm']:,.0f} TPM = **{x['tpm_ratio']:.1f}x the limit**, while cost is only ${x['usd_day']:.2f}/day (${s2['cost_mean']:.5f}/query) and latency needs just {x['concurrency']:.2f} concurrent workers (mean {s2['lat_mean']/1000:.1f}s/query).

Assumptions (edit and re-run `python cost_report.py --today-per-min N --write`): today's volume = {a.today_per_min:g}/min (not measured — the log has no production traffic); limits = Groq on-demand 8,000 TPM / 30 RPM for gpt-oss-120b (`obs.RATE_LIMIT_*`). Token mean from {s2['n']} real v2 runs.
"""
    print(md); print(ten)
    if a.write:
        open("week11/cost_by_stage.md", "w").write(md)
        open("week11/tenx.md", "w").write(ten)

if __name__ == "__main__":
    main()
