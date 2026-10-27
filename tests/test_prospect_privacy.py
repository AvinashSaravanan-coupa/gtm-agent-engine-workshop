import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langsmith import tracing_context

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        profiles.start()
        self.addCleanup(profiles.stop)
        self.prospect_id = next(iter(data_service.PROSPECTS))
        self.record = copy.deepcopy(data_service.PROSPECTS[self.prospect_id])

    def assert_private(self, value):
        serialized = json.dumps(value)
        for field in ("billing_qualification", "tax_id", "card_on_file",
                      "date_of_birth", "credit_check_ref"):
            self.assertNotIn(field, serialized)
        for value in self.record["billing_qualification"].values():
            self.assertNotIn(str(value), serialized)

    def test_contact_allowlist_and_email_flow(self):
        record = {**self.record, "unexpected_field": "not a contact field"}
        with patch.dict(data_service.PROSPECTS, {self.prospect_id: record}):
            result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        expected = {k: self.record[k] for k in (
            "name", "email", "disqualified", "annual_revenue", "enrichment_source"
        )}
        expected["prospect_id"] = self.prospect_id
        self.assertEqual(result, {"prospect": expected, "found": True})
        sent = agent.send_prospect_email.func(
            prospect=result["prospect"], subject="Hello", body="Let's connect.",
            runtime=SimpleNamespace(config={"metadata": {"user_id": data_service.REP_IDS[0]["rep_id"]}}),
        )
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["to"], self.record["email"])
        self.assert_private([result, sent])

    def test_new_profile_is_sanitized_before_save(self):
        runs = []
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        with tracing_context(enabled="local"):
            data_service.get_profile_from_db(
                self.prospect_id, langsmith_extra={"on_end": runs.append}
            )
        expected = {k: v for k, v in self.record.items() if k != "billing_qualification"}
        expected["prospect_id"] = self.prospect_id
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(data_service._PROFILES[self.prospect_id], expected)
        self.assert_private(result)
        self.assertEqual(runs[0].outputs, {"prospect_profile": expected})

    def test_cached_profile_and_traced_lookup_are_sanitized(self):
        cached = {"prospect_id": self.prospect_id, **self.record, "custom_field": "preserved"}
        data_service._PROFILES[self.prospect_id] = cached
        runs = []
        with tracing_context(enabled="local"):
            lookup = data_service.get_profile_from_db(
                self.prospect_id, langsmith_extra={"on_end": runs.append}
            )
        expected = {k: v for k, v in cached.items() if k != "billing_qualification"}
        self.assertEqual(lookup, {"prospect_profile": expected})
        self.assertEqual(runs[0].outputs, lookup)
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assert_private([lookup, runs[0].outputs, result])

    def test_scoring_filters_direct_input_and_completes_profile_flow(self):
        offering = agent.lookup_offering.invoke({"offering_id": next(iter(data_service.OFFERINGS))})
        score = agent.ProspectScore(
            score=75, justification="Suitable fit",
            rubric_breakdown=agent.RubricBreakdown(revenue_fit=80, tech_stack_match=70, segment_fit=75),
        )
        profiles = [
            {"prospect_id": self.prospect_id, **self.record},
            {**self.record, "custom_field": "preserved"},
            agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"],
        ]
        for profile in profiles:
            original = copy.deepcopy(profile)
            with self.subTest(has_id="prospect_id" in profile), patch.object(
                agent, "_scoring_llm", Mock()
            ) as model:
                model.invoke.return_value = score
                result = agent.score_prospect.invoke({
                    "prospect_profile": profile, "offering": offering["offering"],
                })
                self.assertEqual(result, score.model_dump())
                messages = model.invoke.call_args.args[0]
                self.assert_private(messages)
                sent_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
                self.assertEqual(sent_profile, {
                    k: v for k, v in original.items() if k != "billing_qualification"
                })
                self.assertEqual(profile, original)
        self.assert_private([offering, profiles[-1], result])

    def test_missing_prospect_behavior(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})


if __name__ == "__main__":
    unittest.main()
