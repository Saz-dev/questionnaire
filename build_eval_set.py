"""
Week 6 — build eval_set.jsonl from REAL traced data.

Every case here is either:
  - one of the 20 traces already open-coded and mode-tagged in taxonomy.md/notes.md
    (Week 5's seeded random sample, seed 20260903), or
  - the 4 traces from the "what are the tools provided with the cb350 hness" A/B
    across retrieval modes, or
  - 2 freshly-run live queries (see their trace_ids below).

No answer text here is invented — everything is pulled from traces.jsonl by
trace_id. `expected_keywords` / `expected_sources` are filled in either from
golden_set.jsonl (cross-referenced by matching question text) or, for the 2 new
spec-lookup cases, from that case's own real (manually verified) answer.

Run: python build_eval_set.py   -> writes eval_set.jsonl
"""
import json
import re
from pathlib import Path

TRACES_PATH = Path("traces.jsonl")
GOLDEN_PATH = Path("golden_set.jsonl")
OUT_PATH = Path("eval_set.jsonl")

# mode_tag values match the 5 named modes in taxonomy.md, plus "clean" for
# no-known-issue control cases.
CASES = [
    # --- from the Week 5 seeded random sample (seed 20260903) ---
    ("a2b631ae-97bd-4143-ac40-952357212020", "clean", False, True),
    ("4ee731cf-799c-4578-93d1-eb7768546b9b", "model-conflation", False, True),
    ("2afeb46d-5e84-4086-833c-ed2b57c24a2a", "model-conflation", False, True),
    ("e8c6a0d4-fc95-4870-88dd-08dd4c0996e9", "fails-to-use-info", True, True),   # regression #1
    ("499247f3-cce9-4dd5-a1d8-07afd1cfa500", "appropriate-abstention", False, True),
    ("a92095c8-1685-4508-ae35-6e9a761c2c95", "citation-evidence-mismatch", False, True),
    ("66d5a1f0-8aa7-46a6-ba95-765f89dfb249", "fails-to-use-info", False, True),
    ("4caa4e8f-27c3-4d0d-aca5-1771dfae7b47", "appropriate-abstention", False, True),
    ("9cf27693-cf3c-4aaa-8e84-602b5a54e9b2", "unsupported-diagnosis", False, True),
    ("6016cbcd-8861-49f4-9c6f-462d7f41c436", "model-conflation", False, True),
    ("21f2bee7-d711-4a10-9fb0-6dcc5f32a77c", "appropriate-abstention", False, True),
    ("b275bf1c-4e2b-4a0c-a4b4-b4e62b95daab", "citation-evidence-mismatch", False, True),
    ("fae4765b-1910-48b7-bb7e-9176839fb311", "model-conflation", False, True),
    ("b3b17def-fdc5-4313-95f1-5e1d8de6953d", "model-conflation", True, True),    # regression #2
    ("1f5f6a2c-4ad1-49ba-9f69-145e6e42acd9", "appropriate-abstention", False, False),  # redundant w/ 499247f3, excluded from judge/label set
    ("ee8960bb-24f0-42bf-9314-fe15bde26665", "fails-to-use-info", False, True),
    ("42c6302c-ea0f-4fdb-8096-efacc99a4a7e", "model-conflation", False, True),
    ("e82a9259-0219-4a95-abb7-f998deced27c", "model-conflation", False, True),
    ("ed607834-1fcd-4f9a-b499-d783e1a5127f", "model-conflation", False, True),
    ("9669b81b-2bbf-45ad-b1f9-5eef5bbb1b4e", "model-conflation", False, True),
    # --- "tools provided with the cb350 hness" A/B across retrieval modes ---
    ("e9e05d87-9b29-4fed-9bdd-bf21a27b8b20", "fails-to-use-info", False, True),
    ("f8bc4f3d-89a2-45f3-b063-c0b119cf16e6", "fails-to-use-info", False, True),
    ("ba359e0d-4a70-4e78-bafd-97ed8a0dbb1a", "fails-to-use-info", False, True),
    ("cc0ac683-f920-49cd-8825-3fdc98cb0668", "clean", False, True),
    # --- 2 fresh live queries, run today for this eval set ---
    ("6d4448f0-66e7-4a5a-9f3b-46f42983618f", "model-conflation", False, True),
    ("004acaab-4a3a-49cd-b624-0aaf49173929", "clean", False, True),
]

# manual expected values for the 2 fresh cases + the tools-provided cases
# (verified by reading the actual manual chunk text, same way golden_set.jsonl
# was built — see week5 notes.md / the tools-provided investigation).
MANUAL_EXPECTED = {
    "004acaab-4a3a-49cd-b624-0aaf49173929": {  # wheelbase of the CB300R
        "expected_keywords": ["1,352 mm", "1352 mm"],
        "expected_sources": ["OMCB300R.pdf"],
    },
    "cc0ac683-f920-49cd-8825-3fdc98cb0668": {  # tools provided (hyde) - correct answer
        "expected_keywords": ["spark plug wrench", "screwdriver", "fuse puller", "allen"],
        "expected_sources": ["OMHness.pdf"],
    },
    "e9e05d87-9b29-4fed-9bdd-bf21a27b8b20": {  # tools provided (hybrid_mmr) - same expected fact
        "expected_keywords": ["spark plug wrench", "screwdriver", "fuse puller", "allen"],
        "expected_sources": ["OMHness.pdf"],
    },
    "f8bc4f3d-89a2-45f3-b063-c0b119cf16e6": {
        "expected_keywords": ["spark plug wrench", "screwdriver", "fuse puller", "allen"],
        "expected_sources": ["OMHness.pdf"],
    },
    "ba359e0d-4a70-4e78-bafd-97ed8a0dbb1a": {
        "expected_keywords": ["spark plug wrench", "screwdriver", "fuse puller", "allen"],
        "expected_sources": ["OMHness.pdf"],
    },
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip().rstrip("?")


def load_golden_by_question():
    out = {}
    with GOLDEN_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            out[norm(g["question"])] = g
    return out


def load_traces_by_id():
    out = {}
    with TRACES_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["trace_id"]] = d
    return out


def main():
    golden = load_golden_by_question()
    traces = load_traces_by_id()

    rows = []
    for trace_id, mode_tag, regression, judge_eligible in CASES:
        t = traces[trace_id]
        g = golden.get(norm(t["question"]))

        expected_keywords = None
        expected_sources = None
        if trace_id in MANUAL_EXPECTED:
            expected_keywords = MANUAL_EXPECTED[trace_id]["expected_keywords"]
            expected_sources = MANUAL_EXPECTED[trace_id]["expected_sources"]
        elif g:
            expected_keywords = g["keywords"]
            expected_sources = ["OMHness.pdf"]  # golden_set.jsonl is pinned to OMHness.pdf chunk_ids

        rows.append({
            "id": f"w6-{len(rows)+1:02d}",
            "trace_id": trace_id,
            "question": t["question"],
            "retrieval_mode": t["mode"],
            "failure_mode_tag": mode_tag,
            "regression_case": regression,
            "judge_eligible": judge_eligible,
            "expected_grounded": mode_tag != "appropriate-abstention",
            "expected_keywords": expected_keywords,
            "expected_sources": expected_sources,
            "evidence_chunk_ids": t["evidence_chunk_ids"],
            "grounded": t["grounded"],
            "answer": t["answer"],
        })

    with OUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} cases to {OUT_PATH}")
    print(f"  regression cases: {sum(r['regression_case'] for r in rows)}")
    print(f"  judge-eligible:   {sum(r['judge_eligible'] for r in rows)}")
    from collections import Counter
    print("  mode distribution:", dict(Counter(r["failure_mode_tag"] for r in rows)))


if __name__ == "__main__":
    main()
