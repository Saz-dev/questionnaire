# Tool count: before -> after (server one -> server one + two)

Counts come from `tools/list` itself, never from our own notes — each
number is the length of the `tools` array the server returned during the
run, captured verbatim in `discovery_before.json`, `discovery_after.json`
and `wire.json` (frame 5).

## Before (server one only) — commit b04bd18

`tools/list` on `policy-docs` returned:

```
result.tools = [
  "search_policy_documents",   # OUR server (policy-docs)
  "get_policy"                 # OUR server (policy-docs)
]
```

**N = 2 tools discovered**, both from our own policy-document search server.

## After (server one + claims-system) — one config block later

`tools/list` on each server returned:

```
policy-docs    -> tools[ search_policy_documents, get_policy ]
claims-system  -> tools[ claim_status, claim_adjuster_notes ]
```

**M = 4 tools discovered**, the previous 2 plus:

* `claim_status`         # third-party claims server
* `claim_adjuster_notes` # third-party claims server

## The one-line summary

**2 tools before -> 4 tools after**: before = [`search_policy_documents`, `get_policy`],
after adds [`claim_status`, `claim_adjuster_notes`] — the agent's code did not change
to see the +2; it re-ran the same `tools/list` against the extra server that the
config change registered.