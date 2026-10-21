"""Week 12 self-tests. No network, no API key. Run: python capstone.py selftest"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from week12 import agent, packet as pk, pii, llm  # noqa: E402
from week12.mcp_host import McpSession, discover  # noqa: E402
from week12.retriever import get_wordings  # noqa: E402

FN = {f["fnol_id"]: f for f in json.load(open(ROOT / "week12/data/fnols.json"))["fnols"]}


class DateFilter(unittest.TestCase):
    def test_2024_loss_never_gets_2025_flood_wording(self):
        w = get_wordings()
        for d in ("2024-01-01", "2024-08-11", "2024-12-31"):
            hits = w.retrieve("flood exclusion flooding surface water", product="household", as_of=d)["hits"]
            self.assertTrue(hits and all(h["version"] == "2023" for h in hits), d)

    def test_cutover_day_is_2025(self):
        hits = get_wordings().retrieve("flood exclusion", product="household", as_of="2025-01-01")["hits"]
        self.assertTrue(all(h["version"] == "2025" for h in hits))

    def test_unfiltered_ranking_reproduces_the_failure(self):
        hits = get_wordings().retrieve("flood exclusion flooding surface water river rising groundwater water entering",
                                       product="household", as_of="2024-08-11", enforce_date=False)["hits"]
        self.assertEqual(hits[0]["version"], "2025")   # the 2025 wording 'reads better' -> ranks first

    def test_server_refuses_to_run_without_a_loss_date(self):
        with McpSession() as s:
            conn = s.conns["policy-wording"]
            res = conn.request("tools/call", {"name": "search_policy_wording",
                                              "arguments": {"query": "flood", "product": "household", "as_of": ""}})
            self.assertTrue(res["isError"])
            self.assertIn("as_of", res["content"][0]["text"])


class NeverSaysCovered(unittest.TestCase):
    def test_guard_replaces_decision_language(self):
        for bad in ("This loss is covered.", "The claim is not covered under flood.", "We will pay in full.",
                    "Approved: eligible for payment."):
            safe, tripped = pk.guard_narrative(bad, pk.SAFE_NARRATIVE)
            self.assertTrue(tripped, bad)
            self.assertEqual(safe, pk.SAFE_NARRATIVE)

    def test_quoted_wording_is_exempt_but_narrative_is_not(self):
        self.assertEqual(pk.decision_words("Facts.\n> We insure loss; flood is not covered unless FE-2025."), [])
        self.assertTrue(pk.decision_words("Facts: flood is not covered.\n> wording"))

    def test_every_fnol_ends_in_referral_with_no_decision_word(self):
        with McpSession() as s:
            for fid, f in FN.items():
                r = agent.run_triage(f, session=s)
                self.assertTrue(r["packet"]["referral"]["referred"], fid)
                self.assertEqual(r["trajectory"][-1]["tool"], "refer_to_adjuster", fid)
                self.assertEqual(pk.decision_words(r["packet"]["text"]), [], fid)

    def test_injection_text_is_flagged_and_not_obeyed(self):
        with McpSession() as s:
            r = agent.run_triage(FN["F-18"], session=s)
        self.assertIn("prompt_injection_text_present", r["packet"]["flags"])
        self.assertEqual(pk.decision_words(r["packet"]["text"]), [])


class ThreeMoneyNumbers(unittest.TestCase):
    def test_flood_household_three_distinct_numbers(self):
        with McpSession() as s:
            m = agent.run_triage(FN["F-01"], session=s)["packet"]["money"]
        self.assertEqual((m["deductible"], m["sub_limit"], m["policy_limit"]), (1500, 15000, 250000))


class RuntimeDiscovery(unittest.TestCase):
    def test_no_mcp_tool_name_is_hardcoded_in_the_agent(self):
        src = (ROOT / "week12/agent.py").read_text()
        for name in ("search_policy_wording", "claim_adjuster_notes", "wording_lookup_v2"):
            self.assertNotRegex(src, rf"[\"']{name}[\"']")
        self.assertNotRegex(src, r"[\"']claim_status[\"']")

    def test_agent_uses_whatever_tool_the_config_advertises(self):
        cfg = {"mcpServers": {"alt": {"command": "python", "args": ["week12/tests/alt_server.py"]}}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(cfg, fh)
        with McpSession(Path(fh.name)) as s:
            r = agent.run_triage(FN["F-01"], session=s)
        names = [t["tool"] for t in r["trajectory"]]
        self.assertIn("wording_lookup_v2", names)
        self.assertNotIn("search_policy_wording", names)
        self.assertEqual({c["endorsement_version"] for c in r["packet"]["citations"]}, {"2023"})

    def test_with_no_wording_server_the_agent_flags_and_still_refers(self):
        cfg = {"mcpServers": {}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(cfg, fh)
        with McpSession(Path(fh.name)) as s:
            r = agent.run_triage(FN["F-01"], session=s)
        self.assertIn("wording_unavailable", r["packet"]["flags"])
        self.assertTrue(r["packet"]["referral"]["referred"])


class TraceAcrossHops(unittest.TestCase):
    def test_one_trace_id_reaches_our_server_and_comes_back(self):
        with McpSession() as s:
            r = agent.run_triage(FN["F-01"], session=s, trace_id="test-trace-0001")
        t = r["trace"]
        self.assertEqual(t["trace_id"], "test-trace-0001")
        self.assertTrue(t["server_echo"] and all(e["echo_trace_id"] == "test-trace-0001" for e in t["server_echo"]))
        log = (ROOT / "week12/runtime/server_log.jsonl").read_text()
        self.assertIn('"trace_id": "test-trace-0001"', log)
        kinds = {s["kind"] for s in t["spans"]}
        self.assertTrue({"tools", "retrieval"} <= kinds)
        self.assertTrue(all("cost_usd" in s and "duration_ms" in s for s in t["spans"]))

    def test_third_party_server_is_marked_as_a_trace_gap(self):
        with McpSession() as s:
            r = agent.run_triage(FN["F-20"], session=s)
        self.assertIn("trace_gap:claims-system", r["packet"]["flags"])


class PiiLine(unittest.TestCase):
    def test_our_mcp_server_returns_and_logs_no_pii(self):
        probe = "Priya Natarajan of 14 Elm Street had a sprained wrist: flood exclusion"
        with McpSession() as s:
            res, _ = s.conns["policy-wording"].call_tool_traced(
                "search_policy_wording", {"query": probe, "product": "household", "as_of": "2024-08-11"}, "pii-probe-1")
        wire = json.dumps(res)
        log = "\n".join(l for l in (ROOT / "week12/runtime/server_log.jsonl").read_text().splitlines() if "pii-probe-1" in l)
        for marker in ("Priya", "Natarajan", "Elm Street", "sprained"):
            self.assertNotIn(marker.lower(), (wire + log).lower())

    def test_traces_never_store_free_text_fnol(self):
        agent.PII_MODE = "hash"
        with McpSession() as s:
            t = agent.run_triage(FN["F-09"], session=s)["trace"]
        self.assertNotIn("Tomasz", json.dumps(t))
        self.assertIn("FNOL-TEXT-WITHHELD", t["question"])

    def test_claimant_fields_from_the_third_party_server_never_reach_the_trace(self):
        with McpSession() as s:
            t = agent.run_triage(FN["F-20"], session=s)["trace"]
        self.assertNotIn("Rose", json.dumps(t).replace("Roseville", ""))

    def test_redaction_catches_cued_names_addresses_injuries_but_not_bare_first_names(self):
        out = pii.redact("This is Dana Whitlock, 9 Mill Road, whiplash. My mate Tomasz saw it.")
        self.assertNotIn("Whitlock", out); self.assertNotIn("Mill Road", out); self.assertNotIn("whiplash", out)
        self.assertIn("Tomasz", out)   # documented known limit (docs/weaknesses.md): why free text is hashed, not redacted


class Intake(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(agent.normalize_loss_date("2024-08-11T14:30:00"), ("2024-08-11", None))
        self.assertEqual(agent.normalize_loss_date("11/08/2024")[0], None)
        self.assertEqual(agent.normalize_loss_date(None)[1], "loss_date_missing")


class PlannerFallback(unittest.TestCase):
    def test_llm_planner_invalid_choice_falls_back_to_rules(self):
        def bad(body):   # model asks for a tool that does not exist
            return {"choices": [{"message": {"tool_calls": [{"function": {"name": "approve_claim", "arguments": "{}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        with McpSession() as s:
            r = agent.run_triage(FN["F-01"], session=s, planner=agent.LLMPlanner(transport=bad))
        self.assertIn("planner_fallback", r["packet"]["flags"])
        self.assertEqual(r["trajectory"][-1]["tool"], "refer_to_adjuster")
        self.assertGreater(r["trace"]["totals"]["tokens_in"], 0)   # the model call is on the trace, costed

    def test_llm_planner_valid_choice_is_used(self):
        seq = iter(["policy_lookup", "coverage_in_force_check", None])
        def ok(body):
            n = next(seq)
            tc = [{"function": {"name": n, "arguments": json.dumps(
                {"policy_no": "HH-10001"} if n == "policy_lookup" else {"policy_no": "HH-10001", "loss_date": "2024-08-11"})}}] if n else []
            return {"choices": [{"message": {"tool_calls": tc}}], "usage": {"prompt_tokens": 50, "completion_tokens": 5}}
        with McpSession() as s:
            r = agent.run_triage(FN["F-01"], session=s, planner=agent.LLMPlanner(transport=ok))
        self.assertEqual([t["tool"] for t in r["trajectory"]],
                         ["policy_lookup", "coverage_in_force_check", "refer_to_adjuster"])
        self.assertIn("wording_unavailable", r["packet"]["flags"])   # model stopped early; code still refers + flags it


class Reproducible(unittest.TestCase):
    def test_same_input_same_packet_hash(self):
        with McpSession() as s:
            a = agent.run_triage(FN["F-01"], session=s)["packet"]["sha"]
            b = agent.run_triage(FN["F-01"], session=s)["packet"]["sha"]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
