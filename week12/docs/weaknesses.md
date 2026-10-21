<!-- Soft Suave · The AI Engineering League -->
# Top-3 known weaknesses

> **STATUS: ASSISTANT-DRAFTED — YOU MUST OWN AND DATE THIS.** The brief grades the weaknesses you wrote down on Wednesday *before anyone asked*.
> I cannot supply that provenance. Below are the weaknesses I would write down from building and testing the system. Rewrite them in your own words,
> change the ranking if you disagree, and put your real date here: **dated: ____ (written before the review, by: ____)**.

1. **Peril classification is a keyword scorer, and a wrong peril silently produces plausible wrong money.**
   Evidence: w12-35/36 (traces `813a0d15…`, `dbf597fe…`) — fire answered with flood figures. Fixed for those cases by count-based scoring, but it is still lexical:
   "the storm brought the tree down onto the conservatory and then the drains backed up" has two perils and gets one. Nothing in the packet tells the adjuster the peril was a guess.
   Next, ranked & costed: (a) emit the top-2 perils with scores and flag when the margin is small — ~1 h, $0; (b) LLM classifier behind the same flag with an eval on ≥30 ambiguous FNOLs — ~4 h, ≈$0.0003/claim.

2. **The suite and the fixes share an author, and 38/38 is on cases the fixes were built against.**
   Evidence: 3 of the 4 cases written *after* the first green run failed; all 3 were then fixed and are green. The honest expectation is that the next 4 unseen cases fail about as often.
   The PII "held-out" markers (3) were also chosen by the person who wrote the redactor. Next: have another person write 15 FNOLs blind; report the first-contact pass rate. ~2 h, human time.

3. **The corpus is synthetic and the judge gate is unmet.**
   26 chunks, 12 policies; the date filter cost 0 hit-rate@3 only because the corpus is tiny (18/18 vs 18/18). Judge agreement was never ≥85% on the committed Week 6 files (it recomputes to 64%/60%, not the documented 84%), and nothing was re-measured on integrated LLM outputs because there is no API key here.
   Next: real wordings through the Week-4 ingestion (parse quality becomes a measured limit, per the brief), re-run the bonus, relabel the judge set.

## Also known (not top-3)
* The third-party claims server cannot echo the trace id (`trace_gap`, accepted).
* `redact()` misses cue-less names; mitigated by not storing free text, not by fixing the regex.
* The MCP wording server has no authentication; any local client can call it. Fine for a stdio lab, not for the "another squad over a network" case.
* Cutover date 2025-01-01 is assumed.
