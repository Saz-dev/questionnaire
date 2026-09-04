<!-- Soft Suave · The AI Engineering League -->
# Week 5 — Notes

## 0. How the sample was drawn

```python
import json, random

trace_ids = sorted(set(json.loads(l)["trace_id"] for l in open("traces.jsonl") if l.strip()))
# 95 unique trace_ids in traces.jsonl at the time this was run

SEED = 20260903  # today's date (2026-09-03), used as-is, not tuned after seeing results
random.seed(SEED)
sample_20 = random.sample(trace_ids, 20)   # the open-coding sample (below)
replay_pick = random.choice(trace_ids)      # continuing the same RNG stream, the 1 trace to replay
```

**Seed: `20260903`**

**20 selected trace_ids** (in the order `random.sample` returned them):

```
a2b631ae-97bd-4143-ac40-952357212020
4ee731cf-799c-4578-93d1-eb7768546b9b
2afeb46d-5e84-4086-833c-ed2b57c24a2a
e8c6a0d4-fc95-4870-88dd-08dd4c0996e9
499247f3-cce9-4dd5-a1d8-07afd1cfa500
a92095c8-1685-4508-ae35-6e9a761c2c95
66d5a1f0-8aa7-46a6-ba95-765f89dfb249
4caa4e8f-27c3-4d0d-aca5-1771dfae7b47
9cf27693-cf3c-4aaa-8e84-602b5a54e9b2
6016cbcd-8861-49f4-9c6f-462d7f41c436
21f2bee7-d711-4a10-9fb0-6dcc5f32a77c
b275bf1c-4e2b-4a0c-a4b4-b4e62b95daab
fae4765b-1910-48b7-bb7e-9176839fb311
b3b17def-fdc5-4313-95f1-5e1d8de6953d
1f5f6a2c-4ad1-49ba-9f69-145e6e42acd9
ee8960bb-24f0-42bf-9314-fe15bde26665
42c6302c-ea0f-4fdb-8096-efacc99a4a7e
e82a9259-0219-4a95-abb7-f998deced27c
ed607834-1fcd-4f9a-b499-d783e1a5127f
9669b81b-2bbf-45ad-b1f9-5eef5bbb1b4e
```

---

## 1. Open-coded observations (one honest sentence per trace, zero fixes applied)

1. **`a2b631ae`** — "what is the tyre pressure for the cb300r" — the question names a specific bike and the answer correctly gave only that bike's front/rear kPa figures, citing only `OMCB300R.pdf`, even though the retrieval pool also contained Hness/Hornet chunks.
2. **`4ee731cf`** — "What fuel octane should I use?" — the question doesn't name a bike, and the answer states "RON 91 or higher" sourced only from `OMHornet20.pdf`, with no mention that this came from one specific model's manual.
3. **`2afeb46d`** — "What is the minimum tread depth for the front tyre?" — no bike named; the answer states 1.5 mm and cites both `OMCB300R.pdf` and `OMHness.pdf` as if they were one source, which happens to agree here.
4. **`e8c6a0d4`** — "What battery does the CB350 use?" — all 3 retrieved chunks are from `OMHness.pdf` (the CB350 manual, cosine 0.557, above the 0.35 threshold), yet the answer says "I don't know... the provided documents do not contain information about the battery type."
5. **`499247f3`** — "What is the capital of France?" — best cosine 0.220 (below 0.35), and the app returned the fixed refusal message without calling the LLM.
6. **`a92095c8`** — "How do I check the clutch lever freeplay?" — `evidence_chunk_ids` includes a chunk from `OMHness.pdf`, but the answer's Sources section only names `OMHornet20.pdf` and `OMCB300R.pdf`, omitting the third manual it actually used.
7. **`66d5a1f0`** — "what oil does the cb300r use" — the question names CB300R, retrieval returned an `OMCB300R.pdf` chunk (cosine 0.522, above threshold), and the answer still says "I don't know... no oil information is present."
8. **`4caa4e8f`** — "How do I bleed the front brakes?" — the answer says "I don't know," and the retrieved chunks are about front-wheel removal, not brake bleeding.
9. **`9cf27693`** — "clutch feels heavy lately" — a one-line vague complaint gets a specific causal answer ("using non-MA oil damages the assist-slipper clutch system and causes a heavy lever"), stated as if it already diagnosed this rider's specific bike, when the manual passage is a general warning, not a diagnosis of this symptom.
10. **`6016cbcd`** — "What is the procedure to start the bike after a long storage period?" — no bike named; the answer merges storage-inspection and starting steps quoted from both `OMHness.pdf` and `OMCB300R.pdf` into one combined procedure.
11. **`21f2bee7`** — "what is the top speed of the CB350" — best cosine 0.459 (above threshold, so the LLM was called), and the answer correctly says "I don't know... the provided context does not contain any information about the top speed."
12. **`b275bf1c`** — "What is the clutch lever freeplay?" — `evidence_chunk_ids` includes an `OMCB300R.pdf` chunk, but the answer's Sources section cites `OMHornet20.pdf` twice and never mentions `OMCB300R.pdf`.
13. **`fae4765b`** — "headlight seems dim what could be wrong" — the answer combines two different explanations from two different manuals (`OMHness.pdf`'s moisture-in-lens note and `OMHornet20.pdf`'s capacitor-discharge note) into one list without saying these come from different bikes.
14. **`b3b17def`** — "What is the main fuse rating?" — no bike named; the answer states "20 A," sourced from `OMHornet20.pdf`, while this exact question is pinned in `golden_set.jsonl` to `OMHness.pdf::250` with expected value "30 a."
15. **`1f5f6a2c`** — "what is the capital of France" (mode=`rerank`) — best cosine 0.190 (below threshold), correctly refused without an LLM call, same behavior as trace 5 under a different retrieval mode.
16. **`ee8960bb`** — "What is the procedure for replacing the air filter?" — the answer says "I don't know," while a full-corpus scan shows the phrase "air cleaner element" (the manual's own wording, different from "air filter") appears in 8 chunks that weren't retrieved for this question.
17. **`42c6302c`** — same question and mode as trace 2 ("What fuel octane should I use?"), run about an hour earlier — identical retrieved chunk_ids and near-identical answer text, still with no bike named.
18. **`e82a9259`** — "bike won't start in the morning" — no bike named; the answer follows the literal "if engine will not start" procedure quoted from both `OMHness.pdf` and `OMHornet20.pdf`, which happen to state the same steps.
19. **`ed607834`** — same question as trace 18, run about an hour earlier — same retrieved chunk_ids, same procedure, reworded slightly differently by the LLM.
20. **`9669b81b`** — "How do I check the engine oil level?" — no bike named; the answer merges near-identical checking-oil-level steps quoted from all three manuals (`OMCB300R.pdf`, `OMHornet20.pdf`, `OMHness.pdf`) into one 5-step list.

---

## 2. Replay evidence

**Replayed trace_id (chosen by `random.choice` continuing the seeded stream above): `02de760a-32df-4bcd-aa8d-13ff923b9c8e`**
Question: *"What is the tyre pressure for the front when riding alone?"*, mode `rerank`, original timestamp `2026-08-26T12:40:22Z`.

The trace itself does **not** store the chunk text or the exact prompt sent to the LLM — only
`chunk_id`, `source`, `cosine`/`score`, and the final `answer`. So this replay reconstructs what the
trace's own fields allow (question + mode) and re-runs the live pipeline through the actual
`/api/retrieve` and `/api/query` endpoints (the same code path `rag.answer()` uses), rather than
claiming a byte-for-byte deterministic replay from the trace alone.

**Retrieval — reproduced exactly:**

| | Original `evidence_chunk_ids` | Replayed hits (via `/api/retrieve`) |
|---|---|---|
| 1 | `OMCB300R.pdf::270` | `OMCB300R.pdf::270` |
| 2 | `OMHness.pdf::247` | `OMHness.pdf::247` |
| 3 | `OMHornet20.pdf::272` | `OMHornet20.pdf::272` |

Same 3 chunk_ids, same order, same cosine values — retrieval is deterministic given an unchanged index.

**Generation — original vs replayed:**

Original answer (from the trace):
> 1. The front tyre pressure when riding alone (driver only) is:
>    - **OMCB300R:** 200 kPa (2.00 kgf/cm², 29 psi)【OMCB300R.pdf】
>    - **OMHness:** 200 kPa (2.00 kgf/cm², 29 psi)【OMHness.pdf】
>    - **OMHornet20:** 175 kPa (1.75 kgf/cm², 25 psi)【OMHornet20.pdf】

Replayed answer (via `/api/query`, new trace_id `6ac90170-ea1c-4048-8e0e-ce7d4c981123`, `2026-09-03T06:21:08Z`):
> 1. Front tyre pressure when riding alone (driver only):
>    * OMCB300R – 200 kPa (2.00 kgf/cm², 29 psi)【OMCB300R.pdf†L1-L4】
>    * OMHness – 200 kPa (2.00 kgf/cm², 29 psi)【OMHness.pdf†L1-L4】
>    * Hornet 20 – 175 kPa (1.75 kgf/cm², 25 psi)【OMHornet20.pdf†L1-L4】

**Match:** all 3 numbers and all 3 sources are identical. **Mismatch:** exact wording and citation
markup differ (`【OMCB300R.pdf】` vs `【OMCB300R.pdf†L1-L4】`, "OMHornet20" vs "Hornet 20") — the LLM
call (`temperature=0.2`, no fixed random seed passed to the Groq API) is not byte-for-byte
deterministic even when the retrieved context is identical.

**What could not be reconstructed from the trace alone:**
- The exact prompt text sent to the LLM (`RAG._build_prompt` output) — not stored in the trace; had
  to be rebuilt from live chunk text pulled by `chunk_id`, which only works because the index hasn't
  changed since the trace was written. If the corpus were re-indexed, this chunk text would be gone.
- The reranker's raw scores for the full 50-candidate pool — the trace's `retrieval_score` field is
  `null` for every hit even when `mode` is `rerank`/`hybrid_rerank` (this is a real gap in `tracing.py`'s
  `make_trace()`: it reads `h.get("retrieval_score")` off the **candidate pool** entries, but the
  cross-encoder's `retrieval_score` field is only set on the reranked `hits` after `rerank()` runs —
  so the field is always null in a trace, regardless of mode).
- Any generation config beyond `temperature` (top_p, exact API call parameters) — not logged at all.

## 3. Redaction-before-write confirmation

**Confirmed in code, not just inferred from the data.** `tracing.py`'s `make_trace()` calls
`redact_pii(question)` and `redact_pii(result["answer"])` to build the `question`/`answer` fields of
the trace dict, and `write_trace()` (which appends to `traces.jsonl`) is only ever called afterward,
in `api.py` and `generate_traces.py`, on the dict `make_trace()` already returned. There is no code
path that writes a trace before redaction runs. (This corrects the earlier draft report, which
flagged this as "not enough evidence" — that was true of the trace *data* alone, but reading
`tracing.py:21-24, 46-60` settles it from the source.)

## 4. Why a public benchmark would miss the top-3 modes here

A public RAG/QA benchmark scores answers against a single expected string for a self-contained
question, so it would never catch Mode 2 — an app that answers a real question with a real, correctly
sourced number from the *wrong* one of several near-identical documents looks exactly like a correct
answer unless you know there are 3 documents and which one applies. It also wouldn't catch Mode 3,
since it typically doesn't check whether a citation list matches the retrieval evidence that produced
it, only whether the final text is right. And it would score Mode 1 and Mode 4 identically — both are
"I don't know" — even though one is a bug and the other is exactly correct behavior, a distinction
that only shows up by scanning the underlying corpus, which a benchmark's answer key doesn't expose.

## 5. Dated, falsifiable prediction

**Date: 2026-09-03**

> I will attack **Mode 2 — never resolves which of the 3 motorcycles a question is about** (40%,
> 8/20 in this random sample, and the only mode with a provably wrong numeric answer against
> `golden_set.jsonl`) by adding a lightweight model-disambiguation step to `rag.py`: if the question
> doesn't name one of the 3 known models (CB350/H'ness, Hornet 2.0, CB300R) and the top retrieved
> chunks span more than one source PDF, the app should answer per-model explicitly (the way
> `02de760a`'s tyre-pressure trace already does correctly) instead of picking one manual's number and
> presenting it as universal. In the next seeded 20-trace sample, I predict Mode 2 frequency will
> fall from 40% (8/20) to under 15% (3/20), and the specific `main fuse rating` question will no
> longer answer "20 A" without qualifying which bike that's for.

No fix has been applied yet — `taxonomy.md` and `notes.md` were written and are being committed
before any change to `rag.py`.

**Git commit:** `d7b194d18c2f88140f11aa6f1c2f3dd60351197c` ("taxonomy file added") — committed
2026-09-03 12:41:50 +0530, before any Week 6 work (eval set, assertions, judge) began.
