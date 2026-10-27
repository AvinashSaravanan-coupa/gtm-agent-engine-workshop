import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = next(iter(agent.data_service.OFFERINGS))
        self.rep = agent.data_service.REP_IDS[0]
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": self.rep["rep_id"]}})
        self.score = {
            "score": 70.0, "max_score": 100, "justification": "Partial fit",
            "rubric_breakdown": {
                "revenue_fit": 80.0, "tech_stack_match": 60.0,
                "segment_fit": 70.0, "component_max": 100,
            },
        }
        self.scorer = Mock()
        self.scorer.invoke.return_value = agent.ProspectScore(**self.score)
        self.scoring = patch.object(agent, "_scoring_llm", self.scorer)
        self.scoring.start()
        self.addCleanup(self.scoring.stop)

    def test_email_loads_contact_and_appends_runtime_sender_signature(self):
        for rep in agent.data_service.REP_IDS[:2]:
            with self.subTest(rep=rep["rep_id"]):
                runtime = SimpleNamespace(config={"metadata": {"user_id": rep["rep_id"]}})
                result = agent.send_prospect_email.func(self.prospect_id, "Hello", "Let's talk.", runtime)
                contact = agent.data_service.get_prospect_record(self.prospect_id)
                self.assertEqual(result["status"], "sent")
                self.assertTrue(result["message_id"].startswith("msg-"))
                self.assertEqual((result["to"], result["to_name"]), (contact["email"], contact["name"]))
                self.assertEqual((result["from"], result["from_name"]), (rep["email"], rep["name"]))
                self.assertEqual(result["subject"], "Hello")
                self.assertEqual(result["body"], "Let's talk.\n\n" + rep["name"] + "\n" + rep["email"])

    def test_email_missing_prospect_and_email(self):
        missing = agent.send_prospect_email.func("missing", "Hello", "Body", self.runtime)
        self.assertEqual(missing["status"], "failed")
        self.assertFalse(missing["found"])
        self.assertTrue(missing["error"])
        with patch.dict(agent.data_service.PROSPECTS, {"no-email": {"name": "Contact"}}):
            result = agent.send_prospect_email.func("no-email", "Hello", "Body", self.runtime)
        self.assertEqual(result, {"status": "failed", "error": "Prospect record has no email address."})

    def test_score_loads_builds_and_caches_profile_and_offering(self):
        result = agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": self.offering_id})
        self.assertEqual(result, self.score)
        saved = agent.data_service.get_profile_from_db(self.prospect_id)["prospect_profile"]
        self.assertEqual(saved["prospect_id"], self.prospect_id)
        self.assertIn("engagement_history", saved)
        messages = self.scorer.invoke.call_args.args[0]
        self.assertEqual(messages[0]["content"], agent.SCORING_PROMPT)
        self.assertIn(json.dumps(agent.data_service.get_offering(self.offering_id), indent=2), messages[1]["content"])
        self.assertIn(json.dumps(saved, indent=2), messages[1]["content"])
        agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": self.offering_id})
        self.assertIs(agent.data_service.get_profile_from_db(self.prospect_id)["prospect_profile"], saved)

    def test_score_refreshes_cached_tech_stack(self):
        cached = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"]
        cached = {**cached, "tech_stack": ["stale"]}
        agent.data_service.save_profile_to_db(self.prospect_id, cached)
        agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": self.offering_id})
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        profile = json.loads(content.split("\n\nProspect profile:\n")[1])
        self.assertEqual(profile["tech_stack"], agent.data_service.fetch_tech_stack(self.prospect_id))
        self.assertEqual(cached["tech_stack"], ["stale"])

    def test_score_missing_prospect_does_not_invoke_model(self):
        result = agent.score_prospect.invoke({"prospect_id": "missing", "offering_id": self.offering_id})
        self.assertIsNone(result["score"])
        self.assertFalse(result["found"])
        self.assertTrue(result["error"])
        self.scorer.invoke.assert_not_called()

    def test_score_invalid_offering_does_not_invoke_model(self):
        valid = agent.data_service.get_offering(self.offering_id)
        invalid = [None] + [{**valid, field: value} for field, value in (
            ("required_tech_stack", []), ("min_annual_revenue", None), ("description", ""),
        )]
        for offering in invalid:
            with self.subTest(offering=offering):
                with patch.object(agent.data_service, "get_offering", return_value=offering):
                    result = agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": "invalid"})
                self.assertEqual(result, {"score": None, "error": "Cannot score without a valid offering."})
        self.scorer.invoke.assert_not_called()

    def run_graph(self, calls):
        responses = [AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(i)}])
                     for i, (name, args) in enumerate(calls)]
        responses.append(AIMessage(content="Done"))
        results = [ChatResult(generations=[ChatGeneration(message=message)]) for message in responses]
        with patch.object(ChatOpenAI, "_generate", side_effect=results) as model:
            result = agent.run_agent("Complete my request", user_id=self.rep["rep_id"])
        self.assertEqual(model.call_count, len(calls) + 1)
        self.assertEqual(result["reply"], "Done")
        return [json.loads(message.content) for message in result["messages"] if isinstance(message, ToolMessage)]

    def test_email_and_score_execute_in_two_outer_model_calls(self):
        sent = self.run_graph([("send_prospect_email", {"prospect_id": self.prospect_id, "subject": "Hello", "body": "Body"})])
        self.assertEqual(sent[0]["status"], "sent")
        self.assertEqual(sent[0]["from"], self.rep["email"])
        scored = self.run_graph([("score_prospect", {"prospect_id": self.prospect_id, "offering_id": self.offering_id})])
        self.assertEqual(scored, [self.score])
        self.scorer.invoke.assert_called_once()

    def test_identity_tool_remains_available(self):
        result = self.run_graph([("get_current_rep", {})])
        self.assertEqual(result, [{"rep": self.rep, "found": True}])

    def test_score_recovery_and_sequential_update_remain_available(self):
        scored = self.run_graph([
            ("score_prospect", {"prospect_id": "missing", "offering_id": self.offering_id}),
            ("score_prospect", {"prospect_id": self.prospect_id, "offering_id": self.offering_id}),
        ])
        self.assertFalse(scored[0]["found"])
        self.assertEqual(scored[1], self.score)
        updated = self.run_graph([
            ("update_prospect_info", {"prospect_id": self.prospect_id, "technology": "New technology"}),
            ("score_prospect", {"prospect_id": self.prospect_id, "offering_id": self.offering_id}),
        ])
        self.assertTrue(updated[0]["updated"])
        self.assertEqual(updated[1], self.score)


if __name__ == "__main__":
    unittest.main()
