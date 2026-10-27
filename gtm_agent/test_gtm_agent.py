import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import data_service, get_prospect, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Demo invitation",
            body="Please join us for a demo.",
            runtime=self.runtime,
        )

    def assert_blocked(self, prospect):
        with patch("gtm_agent.gtm_agent.uuid.uuid4", side_effect=AssertionError("Email must not be generated")):
            self.assertEqual(self.send(prospect), {
                "status": "blocked",
                "error": "Prospect is disqualified; email not sent.",
            })

    def test_disqualified_source_records_are_blocked(self):
        for prospect_id in ("LEAD-50001", "LEAD-50002", "LEAD-50003", "LEAD-50004", "LEAD-50005"):
            with self.subTest(prospect_id=prospect_id):
                prospect = get_prospect.invoke({"prospect_id": prospect_id})["prospect"]
                self.assertIs(prospect["disqualified"], True)
                self.assert_blocked(prospect)

    def test_supplied_eligibility_cannot_override_source_record(self):
        prospect = get_prospect.invoke({"prospect_id": "LEAD-50001"})["prospect"]
        self.assert_blocked({**prospect, "disqualified": False})

    def test_missing_source_record_is_blocked(self):
        self.assert_blocked({"prospect_id": "LEAD-unknown", "email": "prospect@example.com"})

    def test_missing_prospect_id_is_blocked(self):
        self.assert_blocked({"email": "prospect@example.com", "disqualified": False})

    def test_eligible_prospect_is_sent(self):
        prospect = get_prospect.invoke({"prospect_id": "LEAD-12853"})["prospect"]
        self.assertIs(prospect["disqualified"], False)
        result = self.send(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(len(result["message_id"]), 16)
        self.assertEqual({k: v for k, v in result.items() if k != "message_id"}, {
            "status": "sent",
            "to": prospect["email"],
            "to_name": prospect["name"],
            "from": data_service.get_rep("rep_amills")["email"],
            "from_name": data_service.get_rep("rep_amills")["name"],
            "subject": "Demo invitation",
            "body": "Please join us for a demo.",
        })

    def test_eligible_prospect_without_email_still_fails(self):
        self.assertEqual(self.send({"prospect_id": "LEAD-12853"}), {
            "status": "failed", "error": "Prospect record has no email address.",
        })


if __name__ == "__main__":
    unittest.main()
