"""The real client's configuration. Constructed, never called — no network here."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from agentgraph.config import LLMConfig
from agentgraph.llm.client import make_chat_model


class TestMakeChatModel(unittest.TestCase):
    def test_the_config_reaches_the_client(self):
        model = make_chat_model(LLMConfig(base_url="http://x", model="m", temperature=0.1))
        self.assertEqual(model.base_url, "http://x")
        self.assertEqual(model.model, "m")
        self.assertEqual(model.temperature, 0.1)

    def test_the_reply_cap_reaches_the_server(self):
        """A regression guard, not a style preference.

        This is the setting the previous client could not deliver. ChatOpenAI's `max_tokens` is
        aliased to `max_completion_tokens`, which Ollama's /v1 endpoint ignores — measured,
        asking for 8 tokens: 692 were generated. `num_predict` is Ollama's own name for it and
        is honoured, and it is the ceiling on how large a file write_file can emit in one turn.
        """
        self.assertEqual(make_chat_model(LLMConfig(max_tokens=1024)).num_predict, 1024)

    def test_the_context_window_reaches_the_server(self):
        """Also silently dropped by /v1, which is why the Modelfile had to carry it."""
        self.assertEqual(make_chat_model(LLMConfig(num_ctx=16384)).num_ctx, 16384)

    def test_reasoning_is_requested(self):
        """The one flag that makes this the ReAct edition.

        Without it the Thinking model still thinks, but the `<think>` tags arrive INLINE in the
        answer — so the trace prints them as prose and `write_file` is handed a "complete file"
        with a monologue at the top. `/v1` has no concept of this parameter at all, which is
        why the edition could not have been built on ChatOpenAI.
        """
        self.assertTrue(make_chat_model(LLMConfig()).reasoning)

    def test_reasoning_can_be_turned_off_to_reproduce_the_previous_edition(self):
        self.assertFalse(make_chat_model(LLMConfig(reasoning=False)).reasoning)

    def test_top_k_reaches_the_server(self):
        """JetBrains publishes top_k=20 for the Thinking checkpoint; the Instruct edition had none."""
        self.assertEqual(make_chat_model(LLMConfig(top_k=20)).top_k, 20)

    def test_a_leftover_v1_base_url_is_normalised(self):
        """Ollama's native API is not under /v1; an old MELLUM_BASE_URL must still work."""
        model = make_chat_model(LLMConfig(base_url="http://localhost:11434/v1"))
        self.assertEqual(model.base_url, "http://localhost:11434")

    def test_no_credential_is_read_from_the_environment(self):
        """Pointing MELLUM_BASE_URL elsewhere must not leak a real OpenAI credential."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-real-secret"}):
            model = make_chat_model(LLMConfig())
        self.assertNotIn("sk-real-secret", repr(model))

    def test_env_configuration_is_picked_up_when_no_config_is_passed(self):
        with mock.patch.dict(os.environ, {"MELLUM_MODEL": "other-model"}):
            self.assertEqual(make_chat_model().model, "other-model")
