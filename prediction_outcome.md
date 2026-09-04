<!-- Soft Suave · The AI Engineering League -->
# Week 6 — Prediction scored against the outcome

Original prediction (`prediction.txt`, written before judge_v2.txt existed):
> Adding these 2 [w6-19, w6-21] as worked examples will fix that specific
> pattern... and will flip both w6-19 and w6-21 to match my labels... I expect
> agreement_after to land around 22/25 = 88%.

## What actually happened

| | before (v1) | after (v2) |
|---|---|---|
| agreement | 80.0% (20/25) | 84.0% (21/25) |

**Directionally right, mechanistically wrong.** Agreement did go up, but not
because w6-19 or w6-21 flipped — they didn't; both reproduced almost the exact
same reasoning in v2 that they had in v1, despite v2's prompt containing a
worked example built directly from each one's pattern. The one case that DID
flip (w6-14, FAIL -> PASS) was not one I predicted would move, and nothing in
the v2 prompt specifically targets its pattern (see `disagreements_note.md`).

So the prediction's number (88%) overshot the honest result (84%) by missing
that few-shot examples don't reliably generalize even to near-identical live
cases — and it also predicted the wrong *mechanism*, which matters more than
the number being close: if I'd stopped at "agreement went up, prediction
confirmed," I'd have wrongly credited the few-shot examples for a change they
didn't cause.

## What I'd actually try next, given this

Two few-shot examples clearly isn't enough evidence to correct either pattern
reliably — w6-06/w6-13/w6-19 (the paraphrase-vs-new-claim boundary) need more
than one example each, and ideally a clearer rule in the prompt itself ("only
flag information that changes what a reader would believe is true, not
information that's reworded but says the same thing") rather than examples
alone. w6-21's "check every chunk" instruction is already explicit in v2's
prose (not just the example) and still didn't take — that suggests the model
isn't actually re-reading the full context chunk-by-chunk before answering,
which a prompt-only fix may not solve; forcing an explicit intermediate step
("list which chunks discuss <topic implied by the question>, before you
answer") would be a more promising next iteration than another example.
