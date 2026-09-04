<!-- Soft Suave · The AI Engineering League -->
# Week 6 — Disagreement analysis

All 5 judge_v1 vs. hand-label disagreements out of 25 (only 2 required — all 5
included because the pattern across them is the interesting part).

## w6-06 — clutch lever freeplay. Judge: FAIL. Label: PASS. **Judge was right.**
The answer adds "pull the clutch lever and measure how far it moves before the
clutch begins to engage" — a plausible-sounding explanation of what freeplay
means, but not actually present in any of the 3 retrieved chunks, which only
state the 10–20mm figure. On first read I gave this a pass because the number
and the clutch-wear warning were both correctly grounded; I missed that the
*measurement method* itself was invented. The judge caught something I didn't.

## w6-13 — headlight dim. Judge: FAIL. Label: PASS. **Judge was right, more right than it knew.**
The answer claims lens moisture "reduces the amount of light that passes
through." The actual context says the opposite: "This does not impact the
headlight function." The answer doesn't just add an unsupported claim — it
contradicts an explicit sentence in its own evidence. I misread this case on
my first pass and only registered it as a fair paraphrase.

## w6-19 — bike won't start (duplicate). Judge: FAIL both v1 and v2. Label: PASS. **Human was right; the few-shot fix didn't take.**
The answer adds "If it still does not start, repeat the above steps again."
The context's own step d says "wait 10 seconds before trying steps a & b
again" — the answer is a plain paraphrase of an explicit instruction, not a
new claim. judge_v2.txt includes this almost verbatim as "Example B" (a
worked false-FAIL correction) and the judge still failed the live case with
nearly the same reasoning it used in v1. The few-shot example didn't
generalize even to a case built to match it closely.

## w6-21 — tools provided (hybrid_mmr). Judge: PASS both v1 and v2. Label: FAIL. **Human was right; the few-shot fix didn't take either.**
The answer states "No relevant tool list is present in the other documents,"
but one of this trace's own 5 evidence chunks (`OMHornet20.pdf::166`) is
verbatim a tool list. judge_v2.txt's "Example A" is built directly from this
exact pattern (check every chunk before claiming something is absent
everywhere) and the judge still said PASS on the live case, reproducing the
same miss the example was meant to fix.

## w6-14 — main fuse rating. Judge: FAIL (v1) -> PASS (v2). Label: PASS. **Flipped to agree, but not because of anything in the v2 prompt aimed at it.**
Context says "replace a blown fuse with a spare fuse of the same rating"
and "spare main fuse (20 A)" — so "the main fuse rating is 20 A" is a valid
one-step inference from the given text, not an invention. v1 read this too
strictly (spare-fuse-rating != main-fuse-rating, taken literally); v2 agreed
with the more permissive reading. Neither of the two few-shot examples added
in judge_v2.txt was about this case or this pattern — this flip looks like
plain LLM-judge variance, not a fix that generalized from the examples.

Separately, worth flagging for the record: this is the trace already known
(from Week 5 / `taxonomy.md`) to answer the WRONG bike's number — 20 A from
`OMHornet20.pdf` for a question this repo's own `golden_set.jsonl` pins to
30 A from `OMHness.pdf`. Both judge versions PASS it on the "stays within its
own context" criterion, which is doing exactly what it says — that specific
number really is grounded in the context it was given. The wrong-bike problem
is a retrieval/disambiguation failure, a different category (model-conflation
in `taxonomy.md`), and `assertions.py`'s `expected_fact_present` check is what
actually catches it (fails, correctly, since "30 a" never appears in the
answer) — not this judge criterion. Two different checks, doing two different
jobs, both working as designed.

## The pattern across all 5

3 of 5 disagreements (w6-06, w6-13, w6-19) are the same shape: the judge is
sensitive to *any* added phrasing, even a plain paraphrase of something the
context already says. It got 2 of those 3 right (w6-06, w6-13 really do add
new content) and 1 wrong (w6-19 doesn't). 1 disagreement (w6-21) is the
opposite failure — the judge isn't reading every retrieved chunk before
declaring something absent. The v2 few-shot examples targeted exactly w6-19's
and w6-21's patterns and didn't fix either live case, even though agreement
went up overall (see `prediction_outcome.md`) — the improvement came from an
untargeted case (w6-14), not the mechanism the iteration was built around.
