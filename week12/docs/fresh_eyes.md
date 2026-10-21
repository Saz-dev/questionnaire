<!-- Soft Suave · The AI Engineering League -->
# Fresh-eyes note

> **NOT DONE — placeholder.** This needs a person from another squad running the system cold. I cannot produce it, and inventing one would be a fabricated record.

| | |
|---|---|
| Who (name, squad) | _ |
| Date / time-box | _ |
| What they were given | the repo + README only |
| Command they ran first | _ |
| Where they stumbled (verbatim) | _ |
| Time to first working answer | _ |
| What changed in the repo because of it | _ |
| Did they call our MCP server from *their* client? what did it return? | _ (probe to repeat: `python week12/servers/wording_server.py` over stdio; `tools/list`, then `tools/call search_policy_wording {query, product, as_of}`) |

Known likely stumbles to check against (from building it): runs on Python 3.9+ (the repo `.venv` is 3.9.6); `--rerank` needs the venv from `requirements.txt` and a model download; `--llm` / `--planner llm` need `AI_API_KEY`.
