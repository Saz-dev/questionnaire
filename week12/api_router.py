"""FastAPI router for the capstone: clickable citations + the triage call over HTTP.
Mounted by api.py (`app.include_router(week12_router)`)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from week12 import agent
from week12.retriever import get_wordings

router = APIRouter(prefix="/api", tags=["week12"])


class TriageRequest(BaseModel):
    fnol_id: str | None = None
    policy_no: str | None = None
    loss_date: str | None = None
    text: str = ""
    claim_no: str | None = None
    enforce_date: bool = True


@router.get("/week12/wording/{chunk_id:path}")
def wording(chunk_id: str):
    """What a citation link in a triage packet resolves to: the exact clause text, version and dates."""
    for c in get_wordings().chunks:
        if c["chunk_id"] == chunk_id:
            return {k: v for k, v in c.items() if k != "tokens"}
    raise HTTPException(404, f"no wording chunk {chunk_id!r}")


@router.post("/triage")
def triage(req: TriageRequest):
    r = agent.run_triage({"fnol_id": req.fnol_id or "adhoc", "policy_no": req.policy_no, "loss_date": req.loss_date,
                          "text": req.text, "claim_no": req.claim_no}, enforce_date=req.enforce_date)
    p = r["packet"]
    return {**{k: v for k, v in p.items() if k != "text"}, "text": p["text"], "spans": r["trace"]["spans"]}
