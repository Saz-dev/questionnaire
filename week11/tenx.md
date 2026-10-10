# What breaks first at 10x claim volume

**Rate limit (tokens/minute) breaks first — not cost, not latency.** At 3.3 requests/min the provider's 8,000 TPM cap is already hit (mean 2442 tokens/query); assuming today is 1 claim/min, 10x = 10/min = 24,420 TPM = **3.1x the limit**, while cost is only $7.54/day ($0.00052/query) and latency needs just 2.15 concurrent workers (mean 12.9s/query).

Assumptions (edit and re-run `python cost_report.py --today-per-min N --write`): today's volume = 1/min (not measured — the log has no production traffic); limits = Groq on-demand 8,000 TPM / 30 RPM for gpt-oss-120b (`obs.RATE_LIMIT_*`). Token mean from 21 real v2 runs.
