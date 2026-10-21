"""Loss-date-aware retrieval over the wording slice.

Reuses the Week-4 lexical stage unchanged in algorithm: same tokenizer regex and
the same Okapi BM25 parameters (k1=1.5, b=0.75) as vectordb.py. Semantic /
cross-encoder stages are NOT applied to this slice (they need torch + a model
download; vectordb.py imports qdrant at module load). That is a stated limit.

The date filter is CODE: `retrieve(as_of=...)` drops every chunk whose
endorsement version was not in force on that date BEFORE ranking. With
`as_of=None` (or enforce=False) it ranks across all versions, which is the
Week-11-style behaviour that produced the wrong-version failure.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

DATA = Path(__file__).parent / "data"
BM25_K1, BM25_B = 1.5, 0.75
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-:./][a-z0-9]+)*")
_STOP = {"a", "an", "the", "of", "to", "and", "or", "in", "on", "is", "are", "we", "by", "for", "be", "any"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


class Wordings:
    def __init__(self):
        self.versions = json.loads((DATA / "endorsements.json").read_text())["versions"]
        raw = json.loads((DATA / "wordings.json").read_text())["chunks"]
        vmap = {v["version"]: v for v in self.versions}
        self.chunks = []
        for c in raw:
            v = vmap[c["version"]]
            self.chunks.append({
                **c,
                "chunk_id": f"{c['product']}/{c['version']}/{c['clause_id']}",
                "effective_from": v["effective_from"],
                "effective_to": v["effective_to"],
                "tokens": tokenize(c["title"] + " " + c["text"]),
            })
        self._df = Counter(t for c in self.chunks for t in set(c["tokens"]))
        self._avgdl = sum(len(c["tokens"]) for c in self.chunks) / len(self.chunks)

    def version_in_force(self, as_of: str) -> str | None:
        d = parse_date(as_of)
        for v in self.versions:
            lo = parse_date(v["effective_from"])
            hi = parse_date(v["effective_to"]) if v["effective_to"] else date.max
            if lo <= d <= hi:
                return v["version"]
        return None

    def _bm25(self, q: list[str], c: dict) -> float:
        n = len(self.chunks)
        tf = Counter(c["tokens"])
        score = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log((n - self._df[t] + 0.5) / (self._df[t] + 0.5) + 1)
            score += idf * tf[t] * (BM25_K1 + 1) / (
                tf[t] + BM25_K1 * (1 - BM25_B + BM25_B * len(c["tokens"]) / self._avgdl))
        return score

    def retrieve(self, query: str, product: str | None = None, as_of: str | None = None,
                 k: int = 3, enforce_date: bool = True, rerank: bool = False) -> dict:
        """Return {"hits": [...], "resolved_version": ..., "date_filter": bool}."""
        pool = self.chunks
        if product:
            pool = [c for c in pool if c["product"] == product]
        version = None
        if as_of and enforce_date:
            version = self.version_in_force(as_of)
            pool = [c for c in pool if c["version"] == version]
        q = tokenize(query)
        ranked = sorted(((self._bm25(q, c), c) for c in pool), key=lambda x: -x[0])
        pool_n = 10 if rerank else k
        hits = [{**{k2: v for k2, v in c.items() if k2 != "tokens"}, "score": round(s, 4)}
                for s, c in ranked[:pool_n] if s > 0]
        if rerank and hits:
            # Week-4 cross-encoder (reranker.py), applied AFTER the date filter so it can only reorder
            # chunks from the version in force; it cannot bring the other version back.
            from reranker import rerank as ce_rerank  # lazy: needs sentence-transformers + model download
            hits = [{**h, "score": round(h["score"], 4)} for h in ce_rerank(query, hits, top_k=k)]
        return {"hits": hits, "resolved_version": version, "date_filter": bool(as_of and enforce_date),
                "as_of": as_of, "n_candidates": len(pool)}


_singleton: Wordings | None = None


def get_wordings() -> Wordings:
    global _singleton
    if _singleton is None:
        _singleton = Wordings()
    return _singleton
