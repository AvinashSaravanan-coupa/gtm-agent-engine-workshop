import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class GTMAgentTests(unittest.TestCase):
    def test_custom_tools_runtime_and_request_size(self):
        requests = []

        def generate(model, messages, **kwargs):
            requests.append((messages, kwargs))
            if len(requests) == 1:
                reply = AIMessage(content="", tool_calls=[{
                    "name": "get_current_rep", "args": {}, "id": "rep-call",
                    "type": "tool_call",
                }])
            else:
                reply = AIMessage(content="Rep identified.")
            return ChatResult(generations=[ChatGeneration(message=reply)])

        rep = agent.REP_IDS[0]
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), patch.object(
            ChatOpenAI, "_generate", generate
        ):
            result = agent.gtm_agent.invoke(
                {"messages": [{"role": "user", "content": "Look up offering OFFER-10001."}]},
                config={"metadata": {"user_id": rep["rep_id"]}},
            )

        self.assertEqual(agent.agent_model.model_name, agent.MODEL_NAME)
        messages, kwargs = requests[0]
        self.assertEqual({tool["function"]["name"] for tool in kwargs["tools"]}, {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        })
        system_messages = [m.content for m in messages if isinstance(m, SystemMessage)]
        self.assertEqual(system_messages, [agent.SYSTEM_PROMPT])
        tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(json.loads(tool_results[0].content), {"rep": rep, "found": True})
        self.assertEqual(result["messages"][-1].content, "Rep identified.")
        request = json.dumps({
            "messages": [message.model_dump() for message in messages],
            "tools": kwargs["tools"],
        })
        self.assertLess(len(request), 6500)
        print(f"CRM request size (messages and tool schemas): {len(request)} characters")

    def test_run_agent_preserves_invocation_config(self):
        run_id = uuid.uuid4()
        reply = AIMessage(content="Offering found.")
        with patch.object(agent.uuid, "uuid4", return_value=run_id), patch.object(
            agent.gtm_agent, "invoke", return_value={"messages": [reply]}
        ) as invoke:
            result = agent.run_agent(
                "Look up offering OFFER-10001.", user_id="REP-1",
                thread_id="thread-1", environment="staging",
            )
        invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Look up offering OFFER-10001."}]},
            config={
                "run_name": "GTM Assistant", "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-1", "user_id": "REP-1",
                    "environment": "staging", "request_intent": "lookup_offering",
                },
            },
        )
        self.assertEqual(result, {"reply": reply.content, "messages": [reply], "run_id": str(run_id)})


if __name__ == "__main__":
    unittest.main()
