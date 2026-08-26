"""The scripted fake, and the real client's configuration. No network in either."""

from __future__ import annotations

import unittest

from langchain_core.messages import HumanMessage

from agentfix.config import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMConfig
from agentfix.llm.fake import (
    FakeChatModel,
    assistant_invalid_tool_call,
    assistant_text,
    assistant_tool_call,
    assistant_tool_calls,
)


class TestReplyBuilders(unittest.TestCase):
    def test_text_reply_has_no_tool_calls(self):
        reply = assistant_text("done")
        self.assertEqual(reply.tool_calls, [])
        self.assertEqual(reply.text, "done")

    def test_a_tool_call_carries_parsed_arguments_and_an_id(self):
        reply = assistant_tool_call("read_file", {"path": "a.py"}, call_id="c9")
        self.assertEqual(len(reply.tool_calls), 1)
        call = reply.tool_calls[0]
        self.assertEqual(call["name"], "read_file")
        self.assertEqual(call["args"], {"path": "a.py"})
        self.assertEqual(call["id"], "c9")

    def test_usage_is_reported_so_the_graph_has_something_to_add_up(self):
        reply = assistant_tool_call("run_tests", {}, prompt_tokens=556)
        self.assertEqual((reply.usage_metadata or {})["input_tokens"], 556)

    def test_several_calls_get_distinct_default_ids(self):
        reply = assistant_tool_calls([("run_tests", {}), ("list_files", {})])
        self.assertEqual([c["id"] for c in reply.tool_calls], ["call_1", "call_2"])

    def test_duplicate_ids_are_rejected(self):
        """The whole point of an id is to tell two calls apart."""
        with self.assertRaises(AssertionError):
            assistant_tool_calls([("run_tests", {}), ("list_files", {})], call_ids=("c1", "c1"))

    def test_mismatched_id_count_is_rejected(self):
        with self.assertRaises(AssertionError):
            assistant_tool_calls([("run_tests", {})], call_ids=("c1", "c2"))

    def test_an_invalid_call_lands_in_invalid_tool_calls_not_tool_calls(self):
        reply = assistant_invalid_tool_call("read_file", '{"path": ')
        self.assertEqual(reply.tool_calls, [])
        self.assertEqual(len(reply.invalid_tool_calls), 1)


class TestFakeChatModel(unittest.TestCase):
    def test_replies_are_returned_in_order(self):
        llm = FakeChatModel(replies=[assistant_text("one"), assistant_text("two")])
        self.assertEqual(llm.invoke([HumanMessage("go")]).text, "one")
        self.assertEqual(llm.invoke([HumanMessage("go")]).text, "two")

    def test_each_history_is_snapshotted(self):
        llm = FakeChatModel(replies=[assistant_text("a"), assistant_text("b")])
        llm.invoke([HumanMessage("1")])
        llm.invoke([HumanMessage("1"), HumanMessage("2")])
        self.assertEqual([len(c) for c in llm.calls], [1, 2])

    def test_running_off_the_end_of_the_script_is_a_diagnosis(self):
        llm = FakeChatModel(replies=[assistant_text("only one")])
        llm.invoke([HumanMessage("go")])
        with self.assertRaises(AssertionError) as ctx:
            llm.invoke([HumanMessage("go")])
        self.assertIn("more turns than the test scripted", str(ctx.exception))

    def test_bind_tools_keeps_the_same_object_so_state_stays_observable(self):
        llm = FakeChatModel(replies=[assistant_text("x")])
        self.assertIs(llm.bind_tools([]), llm)


class TestLLMConfig(unittest.TestCase):
    def test_defaults_point_at_local_ollama(self):
        config = LLMConfig()
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertEqual(config.model, DEFAULT_MODEL)

    def test_the_api_url_tolerates_a_leftover_v1_suffix(self):
        """An environment set up for the old ChatOpenAI client must not produce /v1/api/ps."""
        self.assertEqual(
            LLMConfig(base_url="http://localhost:11434/v1").api_url,
            "http://localhost:11434",
        )
        self.assertEqual(
            LLMConfig(base_url="http://localhost:11434/").api_url, "http://localhost:11434"
        )

    def test_env_overrides(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"MELLUM_BASE_URL": "http://x/v1", "MELLUM_MODEL": "m"}):
            config = LLMConfig.from_env()
        self.assertEqual(config.base_url, "http://x/v1")
        self.assertEqual(config.model, "m")

    def test_the_config_is_frozen(self):
        with self.assertRaises(Exception):
            LLMConfig().model = "other"  # type: ignore[misc]
