# Support drill — time-to-find

> **To be filled in from the real, squadmate-timed run. Nothing below is measured yet.**
> Squadmate: run `python plant_bad_answer.py` (round 1), then tell the finder only the complaint sentence. Timer: Observability page → "Support drill timer", or `python drill_tools.py start/found`. Answers are checked against the sealed key.

| | Round 1 | Round 2 (bonus) |
|---|---|---|
| Timed by (squadmate) | _name_ | _name_ |
| Time-to-find (mm:ss) | _  :  _ | _  :  _ |
| Slice that found it (time / user / prompt version / input type / cost outlier / output) | _ | _ |
| Correct trace id (checked vs sealed key) | _ | _ |
| Under 5:00? | _ | _ |

**If over 5:00 — the exact log field whose absence cost the time:** _name it here_

**Bonus — the one field/index added between rounds, and the gap it closed:** _field_ · round 1 `mm:ss` → round 2 `mm:ss`

Commands the finder has: `python drill_tools.py find --since yesterday --input-type coverage --says-covered --no-exclusion`, `--text "flood"`, `--user`, `--prompt-version`, `--min-cost`, then `show <trace_id>`; or the Observability page. `trace.json` is exported with `python drill_tools.py export <trace_id> --out week11/trace.json` (the committed `trace.json` is the real captured failure `565fda46…` from the live agent, used to build eval case `w11-01`).
