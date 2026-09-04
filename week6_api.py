"""
Week 6 — assembles the eval-set / assertions / judge-agreement results into one
JSON-able dict for api.py's GET /api/eval/week6. Reads the static artifacts this
week's work produced (eval_set.jsonl, labels_25.json, judge_results_v1/v2.json,
judge_v1/v2.txt, prediction.txt) - it does not call the LLM judge itself; run
`python judge.py v1` / `v2` to (re)generate those first.
"""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from assertions import run_all
from judge import _build_context, _load_chunk_text_by_id

EVAL_SET_PATH = Path("eval_set.jsonl")
LABELS_PATH = Path("labels_25.json")
DRAFT_LABELS_PATH = Path("labels_25_draft_ai.json")
JUDGE_V1_RESULTS = Path("judge_results_v1.json")
JUDGE_V2_RESULTS = Path("judge_results_v2.json")
JUDGE_V1_PROMPT = Path("judge_v1.txt")
JUDGE_V2_PROMPT = Path("judge_v2.txt")
PREDICTION_PATH = Path("prediction.txt")
PREDICTION_OUTCOME_PATH = Path("prediction_outcome.md")

LABEL_CRITERION = (
    "Same single binary criterion as judge_v1.txt: does the answer stay strictly "
    "within what the retrieved context supports, with no invented facts and no "
    "causal/diagnostic claims the context doesn't state? NOT graded here: source "
    "citation presence, citation-vs-evidence match, exact keyword match, or "
    "refusal-when-expected - those are all covered by assertions.py, not this label."
)

ASSERTION_NAMES = (
    "cites_a_source",
    "refusal_matches_expected_grounded",
    "expected_fact_present",
    "citations_match_evidence",
)
# The judge criteria these assertions replaced (for the "N assertions vs N judge
# criteria" count the Week 6 rubric asks for).
REPLACED_JUDGE_CRITERIA = (
    "does the answer cite a source",
    "does the citation list match what evidence was actually used",
    "does a cited number/fact match the manual",
    "did it correctly refuse when it should have",
)
REMAINING_JUDGE_CRITERIA = (
    "does the answer stay within what the retrieved context actually supports "
    "(no invented facts, no unsupported causal/diagnostic claims)",
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _strip_md_preamble(text: str) -> str:
    """Drop the leading HTML comment + '# Title' line these .md files start
    with — the frontend renders this as plain text, not markdown."""
    lines = text.splitlines()
    while lines and (lines[0].startswith("<!--") or lines[0].startswith("# ") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def build_week6_summary() -> dict:
    cases = _load_jsonl(EVAL_SET_PATH)

    labels_doc = _load_json(LABELS_PATH)
    labels_by_id = {}
    if labels_doc:
        labels_by_id = {l["id"]: l for l in labels_doc["labels"]}

    v1_results = _load_json(JUDGE_V1_RESULTS) or []
    v2_results = _load_json(JUDGE_V2_RESULTS) or []
    v1_by_id = {r["id"]: r for r in v1_results}
    v2_by_id = {r["id"]: r for r in v2_results}

    by_mode = defaultdict(lambda: {"pass": 0, "total": 0})
    case_details = []
    regression_cases = []

    for case in cases:
        result = run_all(case)
        mode = case["failure_mode_tag"]
        by_mode[mode]["total"] += 1
        by_mode[mode]["pass"] += result["passed"]

        label = labels_by_id.get(case["id"])
        v1 = v1_by_id.get(case["id"])
        v2 = v2_by_id.get(case["id"])

        detail = {
            "id": case["id"],
            "question": case["question"],
            "failure_mode_tag": mode,
            "retrieval_mode": case["retrieval_mode"],
            "regression_case": case["regression_case"],
            "judge_eligible": case["judge_eligible"],
            "expected_grounded": case["expected_grounded"],
            "assertions": result["checks"],
            "assertions_passed": result["passed"],
            "answer": case["answer"],
            "hand_label": label["verdict"] if label else None,
            "judge_v1_verdict": v1["verdict"] if v1 else None,
            "judge_v2_verdict": v2["verdict"] if v2 else None,
        }
        case_details.append(detail)
        if case["regression_case"]:
            regression_cases.append(detail)

    mode_breakdown = [
        {"mode": mode, "pass_count": c["pass"], "total": c["total"], "rate": c["pass"] / c["total"]}
        for mode, c in sorted(by_mode.items())
    ]
    total_pass = sum(c["pass"] for c in by_mode.values())
    total_n = sum(c["total"] for c in by_mode.values())

    def agreement(results_by_id: dict) -> dict | None:
        if not results_by_id or not labels_by_id:
            return None
        matched = sum(
            1 for i, l in labels_by_id.items()
            if i in results_by_id and results_by_id[i]["verdict"] == l["verdict"]
        )
        total = sum(1 for i in labels_by_id if i in results_by_id)
        if total == 0:
            return None
        return {"rate": matched / total, "matched": matched, "total": total}

    agreement_before = agreement(v1_by_id)
    agreement_after = agreement(v2_by_id)

    disagreements = []
    if labels_by_id and v1_by_id:
        for i, label in labels_by_id.items():
            v1 = v1_by_id.get(i)
            if v1 and v1["verdict"] != label["verdict"]:
                v2 = v2_by_id.get(i)
                disagreements.append({
                    "id": i,
                    "question": next((c["question"] for c in cases if c["id"] == i), ""),
                    "hand_label": label["verdict"],
                    "hand_label_reason": label["reason"],
                    "judge_v1_verdict": v1["verdict"],
                    "judge_v1_reason": v1["reason"],
                    "judge_v2_verdict": v2["verdict"] if v2 else None,
                    "judge_v2_reason": v2["reason"] if v2 else None,
                    "resolved_in_v2": bool(v2 and v2["verdict"] == label["verdict"]),
                })

    return {
        "mode_breakdown": mode_breakdown,
        "overall_pass_rate": total_pass / total_n if total_n else 0.0,
        "overall_pass_count": total_pass,
        "overall_total": total_n,
        "regression_cases": regression_cases,
        "cases": case_details,
        "assertion_names": list(ASSERTION_NAMES),
        "replaced_judge_criteria": list(REPLACED_JUDGE_CRITERIA),
        "remaining_judge_criteria": list(REMAINING_JUDGE_CRITERIA),
        "labels_recorded_at": labels_doc["labeled_at"] if labels_doc else None,
        "agreement_before": agreement_before,
        "agreement_after": agreement_after,
        "disagreements": disagreements,
        "prediction": PREDICTION_PATH.read_text() if PREDICTION_PATH.exists() else None,
        "prediction_outcome": (
            _strip_md_preamble(PREDICTION_OUTCOME_PATH.read_text()) if PREDICTION_OUTCOME_PATH.exists() else None
        ),
        "judge_v1_prompt": JUDGE_V1_PROMPT.read_text() if JUDGE_V1_PROMPT.exists() else None,
        "judge_v2_prompt": JUDGE_V2_PROMPT.read_text() if JUDGE_V2_PROMPT.exists() else None,
    }


# ----------------------------------------------------------------------
# Human labeling — an actual person (not an LLM) grading each answer blind,
# before the judge's verdict is ever shown. This is the real hand-label step
# Week 6 requires; labels_25_draft_ai.json (Claude's own first-pass labels,
# written while building this feature) is kept only as a reference point, not
# as the graded artifact.
# ----------------------------------------------------------------------

def get_label_queue() -> list[dict]:
    """Every judge-eligible case, with its full context text (same evidence the
    judge itself sees) and any label already saved for it, so the UI can build
    a review queue and support resuming. Judge verdicts are deliberately never
    included here - the whole point is that labeling happens blind."""
    cases = _load_jsonl(EVAL_SET_PATH)
    chunk_text_by_id = _load_chunk_text_by_id()
    labels_doc = _load_json(LABELS_PATH)
    labels_by_id = {l["id"]: l for l in labels_doc["labels"]} if labels_doc else {}

    queue = []
    for case in cases:
        if not case.get("judge_eligible"):
            continue
        existing = labels_by_id.get(case["id"])
        queue.append({
            "id": case["id"],
            "question": case["question"],
            "answer": case["answer"],
            "context": _build_context(case.get("evidence_chunk_ids", []), chunk_text_by_id),
            "existing_verdict": existing["verdict"] if existing else None,
            "existing_reason": existing.get("reason") if existing else None,
        })
    return queue


def submit_label(case_id: str, verdict: str, reason: str, labeler: str) -> dict:
    """Upsert one human label into labels_25.json, creating it (with metadata
    matching the schema Week 5/6 write-ups expect) on the first submission."""
    now = datetime.now(timezone.utc).isoformat()
    doc = _load_json(LABELS_PATH)
    if doc is None:
        doc = {
            "labeler": labeler,
            "labeled_at": now,
            "criterion": LABEL_CRITERION,
            "note": (
                "Labeled through the /evals/label UI: each case's question, full "
                "retrieved context, and answer were shown with no judge verdict "
                "visible. Submitted before judge_v1.txt/judge_v2.txt were (re-)run "
                "against this label set."
            ),
            "labels": [],
        }

    labels = {l["id"]: l for l in doc["labels"]}
    labels[case_id] = {"id": case_id, "verdict": verdict, "reason": reason, "labeled_at": now}
    doc["labels"] = list(labels.values())
    doc["last_updated_at"] = now

    total_eligible = sum(1 for c in _load_jsonl(EVAL_SET_PATH) if c.get("judge_eligible"))
    doc["summary"] = {
        "PASS": sum(1 for l in doc["labels"] if l["verdict"] == "PASS"),
        "FAIL": sum(1 for l in doc["labels"] if l["verdict"] == "FAIL"),
        "total": len(doc["labels"]),
        "remaining": total_eligible - len(doc["labels"]),
    }

    LABELS_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc["summary"]
