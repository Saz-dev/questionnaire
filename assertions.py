"""
Week 6 — deterministic assertions.

These replace 3 criteria that used to be judged subjectively by an LLM. Each one
is a plain rule, not a model call, so it's free, instant, and never has an off day.
Criteria intentionally NOT covered here (left to the judge, see judge_v1.txt):
whether the answer stays within what the evidence actually supports, i.e.
hallucination / unsupported-diagnosis-style failures. A regex can't tell whether
a causal claim is supported by the retrieved text — that's the one thing still
worth paying an LLM for.
"""
import re

KNOWN_SOURCES = ("OMHness.pdf", "OMHornet20.pdf", "OMCB300R.pdf")
REFUSAL_RE = re.compile(r"\bi don'?t know\b", re.IGNORECASE)
NAMES_A_MODEL_RE = re.compile(r"\b(cb350|h.?ness|hornet|cb300r)\b", re.IGNORECASE)


def cites_a_source(answer: str) -> bool:
    """A grounded answer must name at least one of the 3 known manuals in a
    Sources section. A refusal doesn't need to cite anything, so it auto-passes."""
    if REFUSAL_RE.search(answer):
        return True
    return any(src in answer for src in KNOWN_SOURCES)


def refusal_matches_expected_grounded(answer: str, expected_grounded: bool) -> bool:
    """If the question should have been answerable, the answer must not be a
    bare refusal; if it should have been refused, it must be."""
    is_refusal = bool(REFUSAL_RE.search(answer))
    return is_refusal != expected_grounded


def _normalize_spaces(s: str) -> str:
    """Collapse any run of whitespace (including the narrow no-break space,
    U+202F, that the LLM's own output uses between numbers and units, e.g.
    "1,352 mm") down to a single regular space, so a keyword written with
    plain spaces still matches."""
    return re.sub(r"\s+", " ", s)


def expected_fact_present(answer: str, expected_keywords: list[str] | None) -> bool | None:
    """At least one expected keyword must appear in the answer (case-insensitive,
    whitespace-normalized). Returns None (not applicable / auto-pass) when no
    expected_keywords are set — e.g. procedural questions with no single
    canonical number to check."""
    if not expected_keywords:
        return None
    text = _normalize_spaces(answer.lower())
    return any(_normalize_spaces(kw.lower()) in text for kw in expected_keywords)


def citations_match_evidence(question: str, answer: str, evidence_chunk_ids: list[str]) -> bool | None:
    """Every manual actually passed into the prompt as evidence should be named
    somewhere in the answer's own Sources section — Week 5's Mode 3 ("Sources
    list omits a chunk that was actually used") is checkable by a rule, no
    judge needed. Not applicable to refusals (nothing to cite), and not
    applicable when the question names one specific bike model: correctly
    ignoring the other 2 manuals' irrelevant chunks in that case is desired
    behavior (see w6-01, "tyre pressure for the cb300r"), not a citation gap."""
    if REFUSAL_RE.search(answer):
        return None
    if NAMES_A_MODEL_RE.search(question):
        return None
    used_sources = {cid.split("::")[0] for cid in evidence_chunk_ids}
    return all(src in answer for src in used_sources)


def run_all(case: dict) -> dict:
    """Run every assertion on one eval_set.jsonl case. Returns per-check results
    plus an overall `passed` bool (None-valued / not-applicable checks don't
    count against the case)."""
    answer = case["answer"]
    checks = {
        "cites_a_source": cites_a_source(answer),
        "refusal_matches_expected_grounded": refusal_matches_expected_grounded(
            answer, case["expected_grounded"]
        ),
        "expected_fact_present": expected_fact_present(answer, case.get("expected_keywords")),
        "citations_match_evidence": citations_match_evidence(
            case["question"], answer, case.get("evidence_chunk_ids", [])
        ),
    }
    applicable = [v for v in checks.values() if v is not None]
    passed = all(applicable) if applicable else True
    return {"checks": checks, "passed": passed}


if __name__ == "__main__":
    import json
    from collections import defaultdict

    by_mode = defaultdict(lambda: {"pass": 0, "total": 0})
    with open("eval_set.jsonl") as f:
        for line in f:
            case = json.loads(line)
            result = run_all(case)
            mode = case["failure_mode_tag"]
            by_mode[mode]["total"] += 1
            by_mode[mode]["pass"] += result["passed"]

    print(f"{'mode':30s} {'pass':>6s} {'total':>6s} {'rate':>7s}")
    for mode, c in sorted(by_mode.items()):
        rate = c["pass"] / c["total"]
        print(f"{mode:30s} {c['pass']:6d} {c['total']:6d} {rate:6.1%}")
