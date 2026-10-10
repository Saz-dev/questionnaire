"""Week 11 HTTP surface: trace explorer, cost by stage, failure->test loop, drill timer."""
import re
import statistics as st
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import cost_report
import drill_tools
import obs

router = APIRouter(prefix="/api")
W11 = Path(__file__).parent / "week11"


class Span(BaseModel):
    name: str
    kind: str
    start_ms: float
    duration_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    attrs: dict = {}


class TraceRow(BaseModel):
    trace_id: str
    ts: str
    source: Optional[str] = None
    user_id: Optional[str] = None
    prompt_version: Optional[str] = None
    input_type: Optional[str] = None
    says_covered: bool
    says_not_covered: bool
    cites_exclusion: bool
    cost_usd: float
    latency_ms: float
    tokens_in: int
    tokens_out: int
    question: Optional[str] = None
    answer: Optional[str] = None


class TraceDetail(BaseModel):
    trace_id: str
    timestamp: str
    source: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    input_type: Optional[str] = None
    prompt_version: Optional[str] = None
    model: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    retrieved_context_ids: list[str]
    tools_called: list[str]
    spans: list[Span]
    totals: dict
    says_covered: bool
    says_not_covered: bool
    cites_exclusion: bool


@router.get("/traces", response_model=list[TraceRow])
def list_traces(since: Optional[str] = None, until: Optional[str] = None, user_id: Optional[str] = None,
                prompt_version: Optional[str] = None, input_type: Optional[str] = None,
                says_covered: Optional[bool] = None, cites_exclusion: Optional[bool] = None,
                text: Optional[str] = None, min_cost: Optional[float] = None,
                source: Optional[str] = None, limit: int = 100, offset: int = 0):
    s, u = drill_tools._parse_since(since)
    try:
        rows = obs.search_traces(since=s, until=until or u, user_id=user_id, prompt_version=prompt_version,
                                 input_type=input_type, says_covered=says_covered, cites_exclusion=cites_exclusion,
                                 text=text, min_cost=min_cost, source=source, limit=min(limit, 500), offset=offset)
    except Exception as exc:  # malformed FTS query
        raise HTTPException(400, f"bad search: {exc}")
    return [TraceRow(**{k: (bool(v) if k.startswith(("says_", "cites_")) else v)
                        for k, v in r.items() if k in TraceRow.model_fields}) for r in rows]


@router.get("/traces/facets")
def trace_facets():
    rows = obs.search_traces(limit=100000)
    return {
        "users": sorted({r["user_id"] for r in rows if r["user_id"]}),
        "prompt_versions": sorted({r["prompt_version"] for r in rows if r["prompt_version"]}),
        "input_types": sorted({r["input_type"] for r in rows if r["input_type"]}),
        "total": len(rows),
    }


@router.get("/traces/cost")
def traces_cost():
    out = []
    for source in ("agent", "rag"):
        versions = {r["prompt_version"] for r in obs.search_traces(source=source, limit=100000)}
        for pv in sorted(v for v in versions if v):
            ts = cost_report.fleet(source, pv)
            if not ts:
                continue
            s = cost_report.summarize(ts)
            out.append({"source": source, "prompt_version": pv, "n": s["n"], "cost_mean": s["cost_mean"],
                        "cost_p95": s["cost_p95"], "latency_mean_ms": s["lat_mean"], "latency_p95_ms": s["lat_p95"],
                        "tokens_mean": s["tok_mean"], "stage_cost": s["stage_cost"], "stage_latency_ms": s["stage_lat"]})
    return {"groups": out, "rate_limit_tpm": obs.RATE_LIMIT_TPM, "rate_limit_rpm": obs.RATE_LIMIT_RPM,
            "tenx_md": (W11 / "tenx.md").read_text() if (W11 / "tenx.md").exists() else None}


@router.get("/traces/{trace_id}", response_model=TraceDetail)
def trace_detail(trace_id: str):
    t = obs.get_trace(trace_id)
    if not t:
        raise HTTPException(404, "trace not found")
    t = obs.normalize_legacy(t)
    return TraceDetail(**{**t, "says_covered": bool(t["says_covered"]), "says_not_covered": bool(t["says_not_covered"]),
                          "cites_exclusion": bool(t["cites_exclusion"])})


# ---- failure -> test loop ---------------------------------------------------
_MODE_RE = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+[\d.]+%$", re.M)
_OVERALL_RE = re.compile(r"^OVERALL\s+(\d+)\s+(\d+)", re.M)
_CASE_RE = re.compile(r"^\s+(w11-\d+) \[(PASS|FAIL)\]", re.M)


def _parse_suite(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    txt = path.read_text()
    m = _OVERALL_RE.search(txt)
    if not m:
        return None
    modes = [{"mode": a, "passed": int(b), "total": int(c)} for a, b, c in _MODE_RE.findall(txt) if a != "OVERALL"]
    return {"file": path.name, "passed": int(m.group(1)), "total": int(m.group(2)), "modes": modes,
            "w11_cases": [{"id": i, "status": s} for i, s in _CASE_RE.findall(txt)], "raw": txt}


def _read(name: str) -> Optional[str]:
    p = W11 / name
    return p.read_text() if p.exists() else None


@router.get("/eval/week11")
def week11_eval():
    return {
        "red": _parse_suite(W11 / "suite_red.txt"),
        "green": _parse_suite(W11 / "suite_green.txt"),
        "green_attempt1": _parse_suite(W11 / "suite_green_attempt1_guard_regressed.txt"),
        "flake_check": _read("flake_check.txt"),
        "prompt_version_notes": _read("prompt_version_notes.md"),
        "cost_by_stage_md": _read("cost_by_stage.md"),
        "tenx_md": _read("tenx.md"),
        "drill_md": _read("drill.md"),
    }


# ---- drill timer -----------------------------------------------------------
class DrillStart(BaseModel):
    finder: str = "me"
    timed_by: str = "squadmate"
    round: int = 1


class DrillFound(BaseModel):
    trace_id: str
    slice: str
    run_id: Optional[str] = None


@router.get("/drill/runs")
def drill_runs():
    return drill_tools.read_runs()


@router.post("/drill/start")
def drill_start(req: DrillStart):
    return drill_tools.start_run(req.finder, req.timed_by, req.round)


@router.post("/drill/found")
def drill_found(req: DrillFound):
    if req.slice not in drill_tools.SLICES:
        raise HTTPException(400, f"slice must be one of {drill_tools.SLICES}")
    try:
        return drill_tools.finish_run(req.run_id, req.trace_id, req.slice)
    except SystemExit as e:
        raise HTTPException(409, str(e))
