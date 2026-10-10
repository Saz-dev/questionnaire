"""Run each live case N times under each prompt version; report pass rate (LLM nondeterminism check)."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "week9_mcp"))
os.chdir(ROOT / "week9_mcp")
import agent, assertions
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
cases = [json.loads(l) for l in open(ROOT / "eval_set.jsonl") if '"live_agent": true' in l]
for pv in ("agent-v1-2026-10-09", "agent-v2-2026-10-10"):
    os.environ["AGENT_PROMPT_VERSION"] = pv
    for c in cases:
        ok = 0
        for i in range(N):
            t = agent.discover_and_run(c["question"], run_id=f"flake-{c['id']}", user_id="eval-flake")
            ok += assertions.run_all({**c, "answer": t["final_answer"] or ""})["passed"]
        print(f"{pv}  {c['id']:7s} {c['failure_mode_tag']:18s} {ok}/{N}")
