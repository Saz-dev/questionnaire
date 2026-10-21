"""Week 12 integrated triage agent: the Week-7 hand-built loop over a tool registry that
merges LOCAL tools with tools DISCOVERED at runtime over MCP (tools/list), behind one
function: run_triage(fnol).

Loop invariants enforced in CODE (not prompt):
  * step budget (MAX_STEPS);
  * refer_to_adjuster is always the terminal action - if the planner stops without it,
    the loop calls it;
  * MCP tools are picked by capability (input-schema shape), never by hardcoded name;
  * third-party MCP output is stripped of claimant fields and PII-redacted before it
    enters state, the packet, or the trace;
  * the narrative passes packet.guard_narrative.
Planners: RulesPlanner (default, deterministic, reproducible) and LLMPlanner (model picks
the next tool; any invalid choice falls back to the rules and is recorded).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import obs  # noqa: E402
from week12 import llm, packet as pk, pii, tools as local  # noqa: E402
from week12.mcp_host import McpSession  # noqa: E402
from week12.retriever import get_wordings  # noqa: E402

MAX_STEPS = 8
DATA = Path(__file__).parent / "data"
RUNTIME = Path(__file__).parent / "runtime"
PII_MODE = os.environ.get("W12_PII_MODE", "hash")  # "redact" = log regex-redacted FNOL text (before); "hash" = sha+len only (after)
INJECTION_RE = re.compile(r"ignore (all )?(previous|prior) instructions|system note|state that this claim", re.I)

PERIL_RULES = [  # order matters: a burst pipe that "caused flooding" is escape of water, not flood
    ("escape_of_water", r"\b(pipe|burst|appliance|leak(ed|ing)?|tank)\b"),
    ("flood", r"\b(flood\w*|rising groundwater|surface water|river|overflow\w*|water (came|entered))\b"),
    ("theft", r"\b(burglar\w*|break-?in|stolen|theft|forced)\b"),
    ("fire", r"\b(\w*fire|chip pan|smoke|alight)\b"),
    ("windscreen", r"\b(windscreen|windshield)\b"),
    ("hail", r"\bhail\w*\b"),
    ("collision", r"\b(collision|collided|crash\w*)\b"),
    ("storm", r"\b(storm|wind|tiles|fence|gate)\b"),
]
PERIL_QUERY = {
    "flood": "flood exclusion flooding surface water river rising groundwater water entering",
    "escape_of_water": "escape of water pipe tank appliance",
    "theft": "theft forcible entry stolen",
    "fire": "fire lightning explosion",
    "windscreen": "windscreen windows breakage",
    "hail": "hail damage",
    "collision": "collision damage vehicle",
    "storm": "storm wind fences gates damage",
}


def classify_peril(text: str) -> str | None:
    """Score each peril by how many times its vocabulary appears; ties go to the earlier rule.
    v1 took the FIRST rule that matched at all, so 'fire ... hoses flooded the downstairs' became flood
    (eval w12-36) and 'storm ... water came through the ceiling' became flood (w12-35)."""
    best, best_n = None, 0
    for peril, rx in PERIL_RULES:
        n = len(re.findall(rx, text or "", re.I))
        if n > best_n:
            best, best_n = peril, n
    return best


ISO_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$")


def normalize_loss_date(raw: str | None) -> tuple[str | None, str | None]:
    """FNOL intake -> tool contract seam. ISO date or ISO datetime -> YYYY-MM-DD. Anything else
    (11/08/2024 is 11 Aug in the UK and 8 Nov in the US) is NOT guessed: returns (None, flag)."""
    if not raw:
        return None, "loss_date_missing"
    m = ISO_DATE_RE.match(raw.strip())
    if not m:
        return None, "loss_date_unparseable_or_ambiguous"
    try:
        from datetime import date
        date.fromisoformat(m.group(1))
    except ValueError:
        return None, "loss_date_unparseable_or_ambiguous"
    return m.group(1), None


def canon_policy_no(raw: str | None) -> str | None:
    """Intake canonicalisation: 'hh-10001 ' -> 'HH-10001' (eval w12-37)."""
    return (raw or "").strip().upper() or None


def data_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(DATA.glob("*.json")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- registry
def build_registry(session: McpSession) -> dict:
    reg = {n: {"kind": "local", "schema": s, "description": fn.__doc__ or "", "fn": fn}
           for n, (fn, s) in local.LOCAL_TOOLS.items()}
    for n, t in session.tools.items():
        reg[n] = {"kind": "mcp", "server": t["server"], "schema": t["schema"], "description": t["description"]}
    return reg


def find_by_props(reg: dict, props: set[str], kind: str = "mcp") -> str | None:
    for n, t in reg.items():
        if t["kind"] == kind and props <= set(t["schema"].get("properties", {})):
            return n
    return None


def find_claim_status_tool(reg: dict) -> str | None:
    for n, t in reg.items():
        if t["kind"] == "mcp" and t["schema"].get("required") == ["claim_number"] \
                and "status" in t["description"].lower():
            return n
    return None


# ---------------------------------------------------------------- planners
class RulesPlanner:
    name = "rules"

    def next(self, st: dict, reg: dict) -> dict | None:
        f = st["fnol"]
        if "policy_lookup" not in st["done"] and f.get("policy_no"):
            return {"tool": "policy_lookup", "args": {"policy_no": f["policy_no"]}}
        if "policy_lookup" in st["done"] and "error" in st["results"]["policy_lookup"]:
            return None
        if not f.get("loss_date"):
            return None
        if "coverage_in_force_check" not in st["done"]:
            return {"tool": "coverage_in_force_check",
                    "args": {"policy_no": f["policy_no"], "loss_date": f["loss_date"]}}
        if f.get("claim_no") and not st.get("claim_checked"):
            tool = find_claim_status_tool(reg)
            st["claim_checked"] = True
            if tool:
                return {"tool": tool, "args": {"claim_number": f["claim_no"]}}
        search = find_by_props(reg, {"query", "product", "as_of"})
        if st["results"]["coverage_in_force_check"].get("in_force") and "wording" not in st["done"]:
            if search and st["peril"]:
                st["done_wording"] = True
                return {"tool": search, "args": {"query": PERIL_QUERY[st["peril"]],
                                                 "product": st["results"]["policy_lookup"]["product"],
                                                 "as_of": f["loss_date"]}, "tag": "wording"}
        return None


class LLMPlanner:
    """Model picks the next tool. Invalid output -> rules fallback (recorded in st['planner_fallbacks'])."""
    name = "llm"
    SYSTEM = ("You are a claims triage planner. Pick the NEXT tool to call, one at a time, to gather: "
              "the policy, whether it was in force on the loss date, and the policy wording for that "
              "loss date. Never decide coverage. When the facts are gathered reply with no tool call.")

    def __init__(self, transport=None):
        self.fallback, self.transport = RulesPlanner(), transport
        self.usage = []

    def next(self, st: dict, reg: dict) -> dict | None:
        schema = [{"type": "function", "function": {"name": n, "description": t["description"][:400],
                                                    "parameters": t["schema"]}}
                  for n, t in reg.items() if n != "refer_to_adjuster"]
        view = {"fnol": {"policy_no": st["fnol"].get("policy_no"), "loss_date": st["fnol"].get("loss_date"),
                         "claim_no": st["fnol"].get("claim_no"), "text": pii.redact(st["fnol"]["text"])},
                "tools_done": st["done"], "peril_hint": st["peril"]}
        t0 = time.perf_counter()
        try:
            msg, usage = llm.chat([{"role": "system", "content": self.SYSTEM},
                                   {"role": "user", "content": json.dumps(view)}], tools=schema,
                                  transport=self.transport)
        except Exception as e:  # noqa: BLE001
            st["planner_fallbacks"].append(f"llm_error:{type(e).__name__}")
            return self.fallback.next(st, reg)
        self.usage.append((usage, (time.perf_counter() - t0) * 1000))
        calls = msg.get("tool_calls") or []
        if not calls:
            return None
        try:
            name = calls[0]["function"]["name"]
            args = json.loads(calls[0]["function"].get("arguments") or "{}")
            assert name in reg and name not in st["done"]
        except Exception:  # noqa: BLE001
            st["planner_fallbacks"].append("invalid_tool_choice")
            return self.fallback.next(st, reg)
        return {"tool": name, "args": args,
                "tag": "wording" if {"query", "product", "as_of"} <= set(reg[name]["schema"].get("properties", {})) else None}


# ---------------------------------------------------------------- execution
def _strip_third_party(name: str, result):
    if isinstance(result, dict):
        result = {k: v for k, v in result.items() if k not in ("claimant", "holder", "address", "notes")}
        if "notes" in (result or {}):
            pass
    return pii.redact_obj(result)


def _call(name: str, args: dict, reg: dict, session: McpSession, trace_id: str, rec: obs.TraceRecorder,
          st: dict) -> dict:
    t = reg[name]
    if t["kind"] == "local":
        t0 = time.perf_counter()
        out = t["fn"](**args)
        rec.add_span(f"tool.{name}", "tools", (time.perf_counter() - t0) * 1000, side="local")
        return out
    conn = session.conns[t["server"]]
    res, rtt = conn.call_tool_traced(name, args, trace_id)
    text = "".join(c.get("text", "") for c in res.get("content", []))
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = {"error": text}
    meta = res.get("_meta") or {}
    if not conn.echoes_trace and t["server"] not in st["trace_gap"]:
        st["trace_gap"].append(t["server"])   # third-party server cannot echo the id: client-side span only
    server_ms = meta.get("server_ms", 0.0)
    st["server_echo"].append({"tool": name, "server": t["server"], "echo_trace_id": meta.get("trace_id")})
    rec.add_span(f"mcp.transport.{name}", "tools", max(rtt - server_ms, 0.0), side="client", server=t["server"],
                 trace_id=trace_id)
    if meta:
        rec.add_span(f"mcp.server.{name}", "retrieval" if "query" in args else "tools", server_ms,
                     side="server", server=t["server"], trace_id=meta.get("trace_id"))
    return _strip_third_party(name, out)


def run_triage(fnol: dict, *, session: McpSession | None = None, trace_id: str | None = None,
               enforce_date: bool = True, planner=None, narrative_llm: bool = False,
               user_id: str = "adjuster-desk") -> dict:
    own = session is None
    session = session or McpSession().__enter__()
    try:
        return _run(fnol, session, trace_id or str(uuid.uuid4()), enforce_date,
                    planner or RulesPlanner(), narrative_llm, user_id)
    finally:
        if own:
            session.__exit__(None, None, None)


def _run(fnol, session, trace_id, enforce_date, planner, narrative_llm, user_id) -> dict:
    rec = obs.TraceRecorder()
    reg = build_registry(session)
    raw_date = fnol.get("loss_date")
    iso_date, date_flag = normalize_loss_date(raw_date)
    fnol = {**fnol, "loss_date": iso_date, "policy_no": canon_policy_no(fnol.get("policy_no"))}
    st = {"fnol": fnol, "done": [], "results": {}, "peril": classify_peril(fnol.get("text", "")),
          "server_echo": [], "trace_gap": [], "planner_fallbacks": [], "claim_checked": False}
    trajectory: list[dict] = []
    flags: list[str] = []

    for _ in range(MAX_STEPS):
        act = planner.next(st, reg)
        if not act:
            break
        name, args = act["tool"], dict(act["args"])
        is_wording = act.get("tag") == "wording"
        if not enforce_date and is_wording:   # BEFORE-fix mode: the unscoped, all-versions search
            out = _naive_search(args, reg, rec)
        else:
            out = _call(name, args, reg, session, trace_id, rec, st)
        key = "wording" if act.get("tag") == "wording" else name
        st["results"][key] = out
        st["done"].append(key)
        trajectory.append({"tool": name, "args": pii.redact_obj(args), "error": bool(isinstance(out, dict) and "error" in out)})

    # --- code-enforced terminal action
    pol = st["results"].get("policy_lookup", {})
    ifc = st["results"].get("coverage_in_force_check", {})
    wording = st["results"].get("wording", {})
    if "error" in pol:
        flags.append("policy_not_found")
    if date_flag:
        flags.append(date_flag)
    if ifc.get("in_force") is False:
        flags.append("policy_not_in_force")
    if st["peril"] is None:
        flags.append("peril_unclassified")
    if INJECTION_RE.search(fnol.get("text", "")):
        flags.append("prompt_injection_text_present")
    if "wording" not in st["results"] and ifc.get("in_force"):
        flags.append("wording_unavailable")
    if "error" in wording:
        flags.append("wording_error")
    if st["planner_fallbacks"]:
        flags.append("planner_fallback")
    for srv in st["trace_gap"]:
        flags.append(f"trace_gap:{srv}")

    hits = [] if "error" in wording else wording.get("hits", [])
    resolved = wording.get("resolved_version") if "error" not in wording else None
    if wording and not enforce_date:
        resolved = None
    citations = []
    for h in hits:
        citations.append({
            "chunk_id": h["chunk_id"], "clause_id": h["clause_id"], "title": h["title"],
            "endorsement_version": h["version"], "effective_from": h["effective_from"],
            "effective_to": h["effective_to"], "resolved_for_loss_date": fnol.get("loss_date"),
            "text": h["text"], "link": f"/api/week12/wording/{h['chunk_id']}"})
    limits = pol.get("limits")
    money = pk.money_block(limits, st["peril"])
    if limits and st["peril"] and money["sub_limit"] is None:
        flags.append("no_sub_limit_for_peril")

    reason = _referral_reason(flags, ifc, citations)
    t0 = time.perf_counter()
    referral = local.refer_to_adjuster(reason, [c["chunk_id"] for c in citations])
    rec.add_span("tool.refer_to_adjuster", "tools", (time.perf_counter() - t0) * 1000, side="local")
    trajectory.append({"tool": "refer_to_adjuster", "args": {"reason": reason}, "error": False})

    p = {
        "trace_id": trace_id, "fnol_id": fnol.get("fnol_id"), "policy_no": fnol.get("policy_no"),
        "loss_date": fnol.get("loss_date"), "peril": st["peril"], "product": pol.get("product"),
        "in_force": ifc.get("in_force"), "in_force_range": ifc.get("range"),
        "wording": {"resolved_version": resolved, "as_of": fnol.get("loss_date") if enforce_date else None,
                    "date_filter": bool(enforce_date), "server_resolved_version": wording.get("resolved_version")},
        "citations": citations, "money": money, "referral": referral, "flags": flags,
    }
    p["narrative"] = pk.build_narrative(p)

    model_used = None
    if narrative_llm and llm.available():
        t0 = time.perf_counter()
        msg, usage = llm.chat([
            {"role": "system", "content": "Write two plain sentences summarising ONLY the facts given. "
                                          "Never state or imply whether the claim is covered, payable, excluded or approved."},
            {"role": "user", "content": json.dumps(pii.redact_obj({k: p[k] for k in ("policy_no", "loss_date", "peril", "in_force", "flags")}))}])
        rec.add_span("generation.narrative", "generation", (time.perf_counter() - t0) * 1000,
                     tokens_in=usage["prompt_tokens"], tokens_out=usage["completion_tokens"],
                     cost_usd=obs.llm_cost(llm.MODEL, usage["prompt_tokens"], usage["completion_tokens"]))
        safe, tripped = pk.guard_narrative(msg.get("content"), p["narrative"])
        if tripped:
            flags.append("guard_triggered")
        p["narrative"] = safe
        model_used = llm.MODEL
    if isinstance(planner, LLMPlanner):
        for usage, ms in planner.usage:
            rec.add_span("generation.planner", "generation", ms, tokens_in=usage["prompt_tokens"],
                         tokens_out=usage["completion_tokens"],
                         cost_usd=obs.llm_cost(llm.MODEL, usage["prompt_tokens"], usage["completion_tokens"]))
        model_used = llm.MODEL

    p["flags"] = flags
    p["sha"] = pk.packet_sha(p)
    p["text"] = pk.render_text(p)

    prompt_version = "triage-rules-v1" if planner.name == "rules" else "triage-llm-planner-v1"
    trace = obs.build_trace(
        question=_loggable_text(fnol.get("text", "")), answer=pk.strip_quoted(p["narrative"]), prompt_version=prompt_version,
        model=model_used, spans=rec.spans, retrieved_context_ids=[c["chunk_id"] for c in citations],
        source="triage", user_id=user_id, tools_called=[t["tool"] for t in trajectory],
        trace_id=trace_id,
        extra={"fnol_id": fnol.get("fnol_id"), "planner": planner.name, "enforce_date": enforce_date,
               "loss_date": fnol.get("loss_date"), "wording_version": resolved,
               "data_hash": data_hash(), "packet_sha": p["sha"], "flags": flags,
               "server_echo": st["server_echo"], "trace_gap": st["trace_gap"], "mcp_servers": session.raw, "trajectory": trajectory})
    return {"packet": p, "trace": trace, "trajectory": trajectory}


def _loggable_text(text: str) -> str:
    """Free-text FNOL is never stored verbatim. 'hash' mode keeps only sha256+len (the Week-11 note-stub
    idea applied to the whole text); 'redact' mode keeps the regex-redacted text (the weaker BEFORE state)."""
    if PII_MODE == "redact":
        return pii.redact(text) or ""
    return f"[FNOL-TEXT-WITHHELD sha256={hashlib.sha256(text.encode()).hexdigest()[:10]} len={len(text)}]"


def _naive_search(args: dict, reg: dict, rec: obs.TraceRecorder) -> dict:
    """BEFORE-fix behaviour, in-process: rank across ALL endorsement versions (no date filter)."""
    t0 = time.perf_counter()
    res = get_wordings().retrieve(args["query"], product=args["product"], as_of=None, k=3, enforce_date=False)
    rec.add_span("retrieval.naive_unfiltered", "retrieval", (time.perf_counter() - t0) * 1000, side="local")
    return {"query": args["query"], "product": args["product"], "as_of": None,
            "resolved_version": None, "date_filter": False,
            "hits": [{k: h[k] for k in ("chunk_id", "clause_id", "title", "version", "effective_from",
                                        "effective_to", "text", "score")} for h in res["hits"]]}


def _referral_reason(flags: list[str], ifc: dict, citations: list[dict]) -> str:
    if "policy_not_found" in flags:
        return "policy not found: adjuster to locate the policy"
    if "loss_date_missing" in flags:
        return "loss date missing: adjuster to obtain it from the claimant"
    if "loss_date_unparseable_or_ambiguous" in flags:
        return "loss date unparseable or ambiguous (e.g. DD/MM vs MM/DD): adjuster to confirm the date"
    if "policy_not_in_force" in flags:
        return "policy not in force on the loss date: adjuster to confirm cover history"
    if "wording_unavailable" in flags or "wording_error" in flags:
        return "policy wording could not be retrieved: adjuster to read the wording"
    return "routine referral: adjuster to assess the cited wording against the facts"
