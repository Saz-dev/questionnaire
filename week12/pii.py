"""PII line for Week 12: claimant names, addresses and injuries must stay out of
logs, traces, packets and anything our MCP server returns.

Two layers, because neither alone is enough:
  1. REGISTRY  - exact values we KNOW (policy holders, addresses from policies.json).
  2. PATTERNS  - cue-based names ("Mr X", "my name is", "this is"), street
                 addresses, injury lexicon, phones, emails.

Known limit (reported, not hidden): a bare first name with no cue ("my neighbour
Tomasz") and an injury word outside the lexicon get through. week12/eval measures
recall on a HELD-OUT marker set that is not in the registry or lexicon.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).parent / "data"

_STREET = r"(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Close|Court|Row|Walk|Way|Race|Drive|Dr|Place)"
ADDRESS_RE = re.compile(rf"\b\d{{1,4}}\s+(?:[A-Z][a-z]+\s+){{1,2}}{_STREET}\b(?:,\s*[A-Z][a-z]+)?")
CUE_NAME_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
    r"|\b(?i:this is|my name is|i am|i'm|name:|claimant|policyholder)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z'-]+)?"
)
INITIAL_NAME_RE = re.compile(r"\b[A-Z]\.\s[A-Z][a-z]{2,}\b")
INJURY_RE = re.compile(
    r"\b(whiplash|back pain|neck pain|sprain\w*|fractur\w*|broken (?:arm|leg|wrist|rib\w*|bone\w*)|"
    r"concussion|laceration|minor cut|bruis\w*|injur(?:y|ies|ed)|hospital\w*|paramedic|ambulance|"
    r"treated for [a-z ]+)\b", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s.-]?){9,13}\d(?!\d)")

_registry: list[str] | None = None


def registry() -> list[str]:
    """Exact PII values known from the data (longest first so substrings don't pre-empt)."""
    global _registry
    if _registry is None:
        vals: set[str] = set()
        for p in json.loads((DATA / "policies.json").read_text())["policies"]:
            for k in ("holder", "address"):
                v = p.get(k)
                if v and v != "unknown":
                    vals.add(v)
                    if k == "holder":
                        vals.update(part for part in v.split() if len(part) > 3)
                    if k == "address":
                        vals.add(v.split(",")[0])
        _registry = sorted(vals, key=len, reverse=True)
    return _registry


def redact(text: str | None) -> str | None:
    if text is None:
        return None
    for v in registry():
        text = re.sub(re.escape(v), "[REDACTED]", text, flags=re.I)
    text = ADDRESS_RE.sub("[REDACTED-ADDRESS]", text)
    text = CUE_NAME_RE.sub("[REDACTED-NAME]", text)
    text = INITIAL_NAME_RE.sub("[REDACTED-NAME]", text)
    text = INJURY_RE.sub("[REDACTED-INJURY]", text)
    text = EMAIL_RE.sub("[REDACTED-EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED-PHONE]", text)
    return text


def leaked(text: str, markers: list[str]) -> list[str]:
    """Which planted markers still appear (case-insensitive) in `text`."""
    low = (text or "").lower()
    return [m for m in markers if m.lower() in low]


def redact_obj(obj):
    """Redact every string inside a JSON-like structure."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    return obj
