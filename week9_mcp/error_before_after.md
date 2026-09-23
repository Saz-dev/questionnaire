# Requirement 5 — docstring-as-prompt + recoverable error, evidenced before/after

## The tool: `search_policy_documents` on OUR server (`servers/policy_server.py`)

The tool's docstring IS its MCP `description` field in `tools/list`, which the host
feeds to the model as the function's description. So editing the docstring edits the
prompt the model reads — that is the "rewrite a docstring as a prompt" move.

### Before (v `before-rewrite`, commit b04bd18)

```python
def search_policy_documents(query: str) -> dict:
    """Search the claims policy documents."""
    ... if not hits:
        return {"error": "Error 3: lookup failed"}
```

tools/list carried the one-line description *"Search the claims policy documents."*
and the no-match path was the generic `{"error": "Error 3: lookup failed"}`.

### After (v `after-rewrite`, committed with this file)

The same docstring is now written as instructions for the model that reads it, and the
no-match path teaches the model the policy vocabulary and when to stop guessing:

```python
def search_policy_documents(query: str) -> dict:
    """Return the policy clause(s) that cover the event and limit the caller
    asks about, so the caller can answer a coverage question from policy
    language instead of from memory.
    WHEN to call: any question about what a policy covers, excludes, pays, or caps.
    HOW to shape the query: ... The policy speaks of water-ingress loss as
    "flood" under Endorsement W ... "water damage" alone does not appear in the corpus.
    ON a recoverable error: read the error, it names the policy vocabulary;
    re-run ONCE with the corrected query. If ... still returns no match,
    report 'not covered under this policy' rather than inventing a figure."""
    ...
    if not hits:
        return {"error": (
            "No policy clause matched {query!r}. The policy frames water-ingress loss
             as FLOODING under Endorsement W: re-run search_policy_documents with a
             query that names the policy term, e.g. 'flood insurance limit' ... If a
             query naming 'flood' still returns no match, answer 'not covered under
             this policy' and do not guess a number.")}
```

Same corpus, same matcher, same second tool (`get_policy`) — only the docstring and
the error copy changed.

## The same failing call in both worlds

`search_policy_documents` with arguments `{"query": "water damage limit"}` — the
user-facing phrasing for "what does the flood rider pay". Before, this call returns
`{"error": "Error 3: lookup failed"}`. After, it returns a recoverable error.

---

## BEFORE transcript (old docstring + `Error 3`)

Real run `traces/req5_before.json`, model `AI_MODEL`, user query:
*"Claim CLM-2024-88120 is pending review for water damage. Find the water damage limit
in the policy documents and tell me whether the $25,000 claim is within it."*

```
LAP 1  claims-system.claim_status({"claim_number": "CLM-2024-88120"})
       -> {"claim_number": "...88120", "status": "in_review", "claimed_amount": 25000, ...}

LAP 2  policy-docs.search_policy_documents({"query": "water damage limit POL-2024-88120"})
       -> {"error": "Error 3: lookup failed"}                     <- the failing call, isError=true

LAP 3  policy-docs.search_policy_documents({"query": "water damage limit"})
       -> {"error": "Error 3: lookup failed"}                     <- same failing call, still nothing to grab

LAP 4  policy-docs.get_policy({"policy_number": "POL-2024-88120"})
       -> {"product": "Dwelling Comprehensive", "cover": "Dwelling structure + Endorsement W flood rider", ...}
          (the policy has a flood rider, but nothing states a dollar limit)

LAP 5  policy-docs.search_policy_documents({"query": "POL-2024-88120 water damage"})
       -> {"error": "Error 3: lookup failed"}

LAP 6  policy-docs.search_policy_documents({"query": "water damage coverage limit"})
       -> {"error": "Error 3: lookup failed"}

FINAL  (null) — max_laps reached with NO answer; the model could not tell a dead
        expression from a coverage gap and never once tried the word "flood".
```

The generic `Error 3` gave the model nothing to correct, so it looped on "water damage"
variants to exhaustion and produced no answer — exactly the "Swallowing the unknown
case into 'Error: lookup failed'" mistake the task calls out.

## AFTER transcript (recoverable error on the SAME failing call)

Real run `traces/req5_drill2.json`, identical call arguments, rewritten server.

User query: *"Look up \"water damage limit\" in the policy documents and tell me what it is."*

```
LAP 1  policy-docs.search_policy_documents({"query": "water damage limit"})
       -> {"error": "No policy clause matched 'water damage limit'. The policy frames
           water-ingress loss as FLOODING under Endorsement W: re-run
           search_policy_documents with a query that names the policy term, e.g.
           'flood insurance limit' or 'flooding covered'. If a query naming 'flood'
           still returns no match, answer 'not covered under this policy' and do not
           guess a number."}                                          <- same call, now recoverable

LAP 2  policy-docs.search_policy_documents({"query": "flood insurance limit"})
       -> {"matches": [{"clause": "flood", "text": "FLOOD INSURANCE - Under Endorsement
           W, ... is covered up to a limit of $10,000 per occurrence. A separate
           deductible of $500 applies."}]}                             <- one correction, success

FINAL  "The policy covers water-ingress loss as flooding under Endorsement W.
        According to the flood-insurance clause, the limit is $10,000 per occurrence,
        with a $500 deductible."
```

Same first call, same arguments (`{"query": "water damage limit"}`). Old behavior:
loops on `Error 3`, burns six laps, no answer. New behavior: the error names the
vocabulary, the model re-queries the correct way, and answers within two laps.

## Bonus view — the docstring-as-prompt pre-empts the error entirely

Real run `traces/req5_after.json` with the SAME user question as the BEFORE run.
The rewritten description reaches the model on lap 1, so it never needs the error path:

```
LAP 1  claims-system.claim_status({"claim_number": "CLM-2024-88120"}) ...
LAP 2  policy-docs.search_policy_documents({"query": "flood insurance limit"})   <- policy vocabulary, straight away
       -> $10,000 limit clause
FINAL  explains the $10,000 Endorsement-W cap, $500 deductible, and that the
       $25,000 claim is NOT within the covered limit.
```

Before: 6 laps, zero answers, four `Error 3`s. After: 3 laps, correct grounded answer.