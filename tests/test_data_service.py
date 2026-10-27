import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service


class ProspectTechnologyUpdateTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "LANGSMITH_TRACING": "false",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        from gtm_agent import gtm_agent
        self.agent = gtm_agent
        self.records = patch.dict(data_service.PROSPECTS, deepcopy(data_service.PROSPECTS))
        self.records.start()
        self.addCleanup(self.records.stop)
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)

    def test_update_persists_stack_and_refreshes_cached_profile(self):
        prospect_id = "LEAD-39002"
        cached = self.agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        self.assertNotIn("Terraform", cached["prospect_profile"]["tech_stack"])

        result = data_service.update_prospect_info(prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(prospect_id))
        rebuilt = self.agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

    def test_scoring_receives_updated_stack_even_with_old_profile(self):
        prospect_id = "LEAD-39002"
        profile = self.agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        self.agent.update_prospect_info.invoke({
            "prospect_id": prospect_id, "technology": "Terraform",
        })
        offering = data_service.get_offering("OFFER-10004")

        with patch.object(self.agent, "_scoring_llm") as scoring:
            self.agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"], "offering": offering,
            })

        messages = scoring.invoke.call_args.args[0]
        sent_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertIn("Terraform", sent_profile["tech_stack"])
        self.assertTrue(set(offering["required_tech_stack"]).issubset(sent_profile["tech_stack"]))

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        data_service.update_prospect_info("LEAD-39002", "Terraform")

        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002").count("Terraform"), 1)

    def test_missing_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("missing", "Terraform"),
            {"updated": False, "found": False},
        )


if __name__ == "__main__":
    unittest.main()
