"""
The three tools shared, verbatim, by both the agent and the workflow.

get_service_ticket and search_manual are the "existing loop" (Requirement
1's starting point). compute_service_cost is the third tool this task
adds. search_manual is NOT mocked — it calls the real bike-manual RAG
app's own vectordb.py/embedder.py/rag.py pipeline (rerank mode, the same
default main.py/api.py use), filtered to the ticket's own manual, and
applies the exact groundedness rule rag.py:MIN_SCORE already uses
(raw cosine of the pre-rerank candidate pool, not the rerank score).
"""

import os
import sys

from data import BIKE_MODEL_TO_MANUAL, CUSTOMER_EXCESS, STRESS_TICKET, TICKETS

_TICKETS_BY_ID = {t["ticket_id"]: t for t in TICKETS}
_TICKETS_BY_ID[STRESS_TICKET["ticket_id"]] = STRESS_TICKET

COVERAGE_STATUS_ENUM = ["covered", "not_covered", "pending_review"]

# --- wire up the REAL RAG app (questionnaire/) as a library ----------------
# vectordb.py/rag.py resolve their storage paths ("documents", "qdrant_storage")
# relative to CWD, so we cd into questionnaire/ once, build/reuse its cached
# index, then cd back — the opened QdrantClient keeps working after that.
_QUESTIONNAIRE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_original_cwd = os.getcwd()
sys.path.insert(0, _QUESTIONNAIRE_DIR)
os.chdir(_QUESTIONNAIRE_DIR)
from embedder import embed_text  # noqa: E402
from rag import RAG, MIN_SCORE, TOP_K, CANDIDATES  # noqa: E402
from reranker import rerank as _rerank  # noqa: E402
from vectordb import build_index  # noqa: E402

_db = build_index()
_rag = RAG(_db, mode="rerank")
os.chdir(_original_cwd)
# ---------------------------------------------------------------------------


def get_service_ticket(ticket_id: str) -> dict:
    ticket = _TICKETS_BY_ID.get(ticket_id)
    if ticket is None:
        return {"error": f"no service ticket on file with id {ticket_id!r}"}
    return {
        "ticket_id": ticket["ticket_id"],
        "bike_model": ticket["bike_model"],
        "mileage_km": ticket["mileage_km"],
        "reported_symptom": ticket["reported_symptom"],
        "technician_notes": ticket["technician_notes"],
        "parts_cost": ticket["parts_cost"],
        "labor_cost": ticket["labor_cost"],
    }


def search_manual(bike_model: str, query: str) -> dict:
    """
    Real retrieval: semantic candidates -> filtered to this bike's manual
    -> reranked -> top TOP_K excerpts. Refuses (grounded=False) below
    MIN_SCORE on the FILTERED pool's raw cosine, same rule rag.py's
    RAG.answer() applies to the whole corpus.
    """
    manual = BIKE_MODEL_TO_MANUAL.get(bike_model)
    if manual is None:
        return {"error": f"no manual on file for bike model {bike_model!r}"}

    pool = _db.search(embed_text(query), top_k=CANDIDATES)
    filtered = [h for h in pool if h["source"] == manual]
    if not filtered:
        return {"grounded": False, "manual": manual, "message": "no matching passage found in this bike's manual"}

    best_cosine = max(h["cosine"] for h in filtered)
    if best_cosine < MIN_SCORE:
        return {
            "grounded": False,
            "manual": manual,
            "best_cosine": round(best_cosine, 3),
            "message": f"no passage in {manual} is a confident enough match (cosine {best_cosine:.2f} < {MIN_SCORE})",
        }

    # Wider than the app's default TOP_K=3: this tool feeds a downstream
    # model that must locate one specific number in a compact spec table,
    # not compose a prose answer, so recall matters more here than it
    # does for rag.py's normal Q&A path.
    top_hits = _rerank(query, filtered, top_k=8)
    return {
        "grounded": True,
        "manual": manual,
        "best_cosine": round(best_cosine, 3),
        "excerpts": [h["text"] for h in top_hits],
    }


def compute_service_cost(parts_cost: float, labor_cost: float, coverage_status: str) -> dict:
    if coverage_status not in COVERAGE_STATUS_ENUM:
        return {"error": f"coverage_status must be one of {COVERAGE_STATUS_ENUM}, got {coverage_status!r}"}
    if coverage_status == "not_covered":
        return {"payout": 0, "coverage_status": coverage_status}
    if coverage_status == "pending_review":
        return {"payout": None, "coverage_status": coverage_status}
    payout = max(parts_cost + labor_cost - CUSTOMER_EXCESS, 0)
    return {"payout": payout, "coverage_status": coverage_status}


TOOL_IMPLS = {
    "get_service_ticket": get_service_ticket,
    "search_manual": search_manual,
    "compute_service_cost": compute_service_cost,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_service_ticket",
            "description": (
                "Retrieve the stored service ticket for one ticket ID: "
                "the bike model, odometer mileage, reported symptom, the "
                "full technician notes text, and the ticket's parts_cost "
                "and labor_cost. Read-only lookup — it does not look "
                "anything up in a manual and does not compute any final "
                "payout amount."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket identifier, e.g. 'TCK-01'.",
                    }
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": (
                "Search this bike model's official manual for one "
                "specific fact or specification (e.g. the recommended "
                "fuel octane, spark plug, torque value, or fluid type) "
                "and report whether the workshop's actual practice "
                "matches it. Does not retrieve ticket records and does "
                "not compute a cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bike_model": {
                        "type": "string",
                        "description": "The bike model on the ticket, e.g. \"Honda H'ness CB350\".",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "The specific spec to look up, e.g. "
                            "'recommended fuel octane' or 'torque for the "
                            "rear axle nut' — use what the technician "
                            "notes actually describe checking, not "
                            "necessarily the reported symptom."
                        ),
                    },
                },
                "required": ["bike_model", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_service_cost",
            "description": (
                "Compute the final amount the warranty pays out for a "
                "single ticket, given its parts cost, its labor cost, "
                "and its decided coverage status. This tool only "
                f"performs the arithmetic (covered pays parts+labor minus "
                f"the ${CUSTOMER_EXCESS} customer excess; not_covered "
                "pays 0) — it never retrieves a ticket or a manual, and "
                "never decides the status itself; the caller must "
                "already know coverage_status from search_manual's spec "
                "comparison."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parts_cost": {"type": "number", "description": "Parts cost for the repair."},
                    "labor_cost": {"type": "number", "description": "Labor cost for the repair."},
                    "coverage_status": {
                        "type": "string",
                        "enum": COVERAGE_STATUS_ENUM,
                        "description": (
                            "'covered' if actual practice matched the "
                            "manual's spec, 'not_covered' if search_manual "
                            "found a deviation, 'pending_review' if that "
                            "could not be determined."
                        ),
                    },
                },
                "required": ["parts_cost", "labor_cost", "coverage_status"],
            },
        },
    },
]

TOOL_DESCRIPTION_DIFF = """\
--- before (2-tool loop, Requirement 1's starting point) ---
  get_service_ticket(ticket_id)
      "Retrieve the stored service ticket for one ticket ID: the bike
       model, odometer mileage, reported symptom, the full technician
       notes text, and the ticket's parts_cost and labor_cost. Read-only
       lookup — it does not look anything up in a manual and does not
       compute any final payout amount."
  search_manual(bike_model, query)
      "Search this bike model's official manual for one specific fact
       or specification (e.g. the recommended fuel octane, spark plug,
       torque value, or fluid type) and report whether the workshop's
       actual practice matches it. Does not retrieve ticket records and
       does not compute a cost."

+++ after (3rd tool added) +++
+ compute_service_cost(parts_cost: number, labor_cost: number,
+                       coverage_status: enum[covered|not_covered|pending_review])
+     "Compute the final amount the warranty pays out for a single
+      ticket, given its parts cost, its labor cost, and its decided
+      coverage status. This tool only performs the arithmetic (covered
+      pays parts+labor minus the $50 customer excess; not_covered pays
+      0) — it never retrieves a ticket or a manual, and never decides
+      the status itself; the caller must already know coverage_status
+      from search_manual's spec comparison."

Why there is no overlap: get_service_ticket owns "what is the ticket"
(I/O: lookup only, zero judgement). search_manual owns "does actual
practice match spec" (I/O: retrieval + comparison judgement, zero
arithmetic). compute_service_cost owns "what do we pay" (I/O: arithmetic
only, zero lookup, zero judgement — it takes coverage_status as an input
rather than deriving it, so it can never re-decide what search_manual
already decided).
"""

if __name__ == "__main__":
    print(get_service_ticket("TCK-01"))
    print(search_manual("Honda H'ness CB350", "recommended fuel octane"))
    print(compute_service_cost(45, 80, "not_covered"))
    print(TOOL_DESCRIPTION_DIFF)
