<!-- Soft Suave · The AI Engineering League -->
# Week 5 — Taxonomy

Seeded random sample of 20 traces out of 95 total in `traces.jsonl` (seed and IDs: see `notes.md`).
Percentages are out of 20. Modes are not mutually exclusive in principle (a trace can show more
than one), but each trace below is counted once, under the single most notable/actionable mode it
showed — secondary issues on the same trace are called out in that trace's sentence in `notes.md`.

| # | Mode | Count | % of 20 | Severity | Example trace_id |
|---|---|---|---|---|---|
| 1 | **Fails to use manual information it actually has** — answers "I don't know" for a question whose fact is confirmed present in the correct manual (verified by a full-corpus text scan, not just by what got retrieved) | 3 | 15% | High — these are concrete spec/procedure questions (battery, oil grade) a rider would actually need answered | `e8c6a0d4-fc95-4870-88dd-08dd4c0996e9` |
| 2 | **Never resolves which of the 3 motorcycles a question is about** — answers a model-unspecified question by silently merging or picking from more than one manual, instead of asking or clearly separating per-bike | 8 | 40% | Low–High — harmless when the value happens to match across bikes, but High when it doesn't (see the fuse example below) | `b3b17def-fdc5-4313-95f1-5e1d8de6953d` |
| 3 | **"Sources" list omits a chunk that was actually used as evidence** — `evidence_chunk_ids` on the trace contains a chunk from a manual never mentioned in the answer's own Sources section | 3 | 15% | Medium — breaks the citation/audit trail the app advertises; a reader can't tell everything the answer actually rested on | `a92095c8-1685-4508-ae35-6e9a761c2c95` |
| 4 | **Appropriate abstention** — correctly says "I don't know" because the fact is genuinely absent from the corpus (confirmed by a full-corpus scan: 0 hits for the underlying term) | 4 | 20% | Not a failure — flagged only because it is indistinguishable from Mode 1 without a corpus scan | `4caa4e8f-27c3-4d0d-aca5-1771dfae7b47` |
| 5 | **Vague symptom converted into a confident, unsupported diagnosis** — a one-line vague complaint gets a specific causal explanation the manual only states as a general warning, not as a diagnosis of the user's actual symptom | 1 | 5% | High — could send a rider toward the wrong maintenance action | `9cf27693-cf3c-4aaa-8e84-602b5a54e9b2` |

3 + 8 + 3 + 4 + 1 = 19. The remaining 1 of 20 (`a2b631ae-97bd-4143-ac40-952357212020`) is a clean
success — the question named a specific bike and the answer correctly cited only that bike's manual
— and isn't counted under any failure mode.

## The one number that matters most

**Mode 2 (model conflation) is the actual dominant failure at 40% (8/20) in this random sample** —
not "fails to use available information," which was reported as the top mode (42.1%, 8/19) in the
earlier draft (`week5_error_analysis_report.md`). That earlier number came from whatever traces
happened to be in the file at the time, not a documented random draw (see "common mistakes" #2 in
`W5-Task-Set-D.md`). Under a real seeded sample, Mode 1 drops to 15% and Mode 2 — barely mentioned
before — turns out to be the biggest problem. This is exactly the failure mode the random-sampling
requirement exists to catch.

## Concrete evidence for Mode 2's severity range

- `b3b17def-fdc5-4313-95f1-5e1d8de6953d` — question "What is the main fuse rating?" (no bike named).
  The app answered **20 A**, sourced from `OMHornet20.pdf`. The repo's own pinned golden-set answer
  for this exact question (`golden_set.jsonl`) is **30 A**, from `OMHness.pdf::250` (the H'ness/CB350
  manual). The app picked a real number from a real manual — just the wrong bike's — and stated it
  as fact with no caveat. This is the only trace in the sample that is provably, numerically wrong
  against ground truth already in the repo, not just ambiguous.
- `9669b81b-2bbf-45ad-b1f9-5eef5bbb1b4e` / `e82a9259-0219-4a95-abb7-f998deced27c` — same never-names-
  the-bike pattern, but harmless here because the underlying procedure text is genuinely identical
  across all three manuals.

## Corpus checks behind Mode 1 vs Mode 4 (so the split isn't a guess)

| Trace | Question | Term checked | Full-corpus hits | Verdict |
|---|---|---|---|---|
| `e8c6a0d4` | What battery does the CB350 use? | "battery" in `OMHness.pdf` | 37 chunks | Mode 1 — info exists, answer said it didn't |
| `66d5a1f0` | what oil does the cb300r use | "engine oil" in `OMCB300R.pdf` | 27 chunks | Mode 1 — info exists, answer said it didn't |
| `4caa4e8f` | How do I bleed the front brakes? | "bleed" anywhere | 0 chunks | Mode 4 — genuinely not in corpus |
| `21f2bee7` | what is the top speed of the CB350 | "top speed" / "maximum speed" in `OMHness.pdf` | 0 chunks | Mode 4 — genuinely not in corpus |
| `ee8960bb` | air filter replacement procedure | "air filter" (4 hits) vs "air cleaner element" (8 hits, different wording) | some hits, different vocabulary | Mode 1, weaker confidence — looks like a vocabulary mismatch between the question's wording and the manual's, not a truly empty corpus |

(Scan method: same idea as `vectordb.py`'s `find_chunk_id`/`corpus_contains` — a lowercase substring
search over every chunk in `qdrant_storage/chunks.json`, independent of any retrieval strategy.)

## Next mode to attack

See `notes.md` §5 for the dated, falsifiable prediction — a change targeting **Mode 2**, since it's
now the confirmed largest and most safety-relevant mode in a real random sample.
