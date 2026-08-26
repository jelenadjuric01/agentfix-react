"""The preflight checks. Every check must return a verdict rather than raising."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from agentfix import doctor
from agentfix.config import BASE_MODEL, LLMConfig
from agentfix.doctor import (
    Check,
    _check_context,
    _check_model_present,
    _check_python,
    _check_sandbox,
    _check_server,
    report,
)
from agentfix.llm.fake import FakeChatModel, assistant_text, assistant_tool_call

CONFIG = LLMConfig()


def patched_json(payload):
    return mock.patch("agentfix.doctor._get_json", return_value=payload)


class TestChecks(unittest.TestCase):
    def test_python_version(self):
        self.assertTrue(_check_python().ok, "the test suite itself requires >= 3.12")

    def test_an_unreachable_server_names_the_fix_and_nothing_else(self):
        """One failure must not be buried under derived failures."""
        with patched_json(None):
            check = _check_server(CONFIG)
        self.assertFalse(check.ok)
        self.assertIn("ollama serve", check.detail)
        self.assertNotIn("pull", check.detail)

    def test_a_reachable_server_passes(self):
        with patched_json({"models": []}):
            self.assertTrue(_check_server(CONFIG).ok)

    def test_the_derived_model_present(self):
        with patched_json({"models": [{"name": f"{CONFIG.model}:latest"}]}):
            self.assertTrue(_check_model_present(CONFIG).ok)

    def test_the_base_model_without_the_derived_one_is_diagnosed_precisely(self):
        """The most confusing failure mode: right weights, wrong context window."""
        with patched_json({"models": [{"name": f"{BASE_MODEL}:latest"}]}):
            check = _check_model_present(CONFIG)
        self.assertFalse(check.ok)
        self.assertIn("ollama create", check.detail)

    def test_no_model_at_all_says_pull_then_create(self):
        with patched_json({"models": []}):
            check = _check_model_present(CONFIG)
        self.assertFalse(check.ok)
        self.assertIn("ollama pull", check.detail)

    def test_too_small_a_context_window_is_caught(self):
        """It does not error — it silently truncates the history mid-run."""
        with patched_json({"models": [{"name": f"{CONFIG.model}:latest", "context_length": 4096}]}):
            check = _check_context(CONFIG)
        self.assertFalse(check.ok)
        self.assertIn("4096", check.detail)
        self.assertIn("ollama create", check.detail)

    def test_a_large_enough_context_passes(self):
        with patched_json(
            {"models": [{"name": f"{CONFIG.model}:latest", "context_length": 16384}]}
        ):
            self.assertTrue(_check_context(CONFIG).ok)

    def test_a_model_that_is_not_loaded_cannot_report_its_context(self):
        with patched_json({"models": []}):
            self.assertFalse(_check_context(CONFIG).ok)

    def test_the_sandbox_check_really_executes_a_test(self):
        """run_tests is the only oracle; if execution is broken every run reports NOT SOLVED."""
        self.assertTrue(_check_sandbox().ok)


class TestReasoningAndToolChecks(unittest.TestCase):
    """The two checks that catch a setup which is broken but still works.

    Both failures leave you with a running agent, which is why nothing else reports them: point
    this project at an Instruct model and every run completes, silently, as the previous
    workshop's Act-only agent.
    """

    def _run(self, reply):
        """Run the check with a scripted model standing in for the real client."""
        fake = FakeChatModel(replies=[reply])
        with mock.patch("agentfix.llm.client.make_chat_model", return_value=fake):
            return {check.name: check for check in doctor._check_reasoning_and_tools(LLMConfig())}

    def test_a_thinking_model_that_calls_tools_passes_both(self):
        checks = self._run(
            assistant_tool_call("calculate", {"expression": "17*23"}, reasoning="17*23 is 391.")
        )
        self.assertTrue(checks["reasoning"].ok)
        self.assertTrue(checks["tool calling"].ok)
        self.assertIn("calculate", checks["tool calling"].detail)

    def test_an_instruct_model_fails_the_reasoning_check_and_is_named_as_such(self):
        checks = self._run(assistant_tool_call("calculate", {"expression": "17*23"}))
        self.assertFalse(checks["reasoning"].ok)
        self.assertIn("Instruct", checks["reasoning"].detail)
        self.assertTrue(
            checks["tool calling"].ok, "it can still act — only the thinking is missing"
        )

    def test_reasoning_arriving_inline_is_diagnosed_separately(self):
        """A different bug with a different fix: the model thinks, but `reasoning=True` is lost.

        Reporting this as "not a Thinking model" would send you off to re-pull an 8 GB model
        that was never the problem.
        """
        checks = self._run(assistant_text("<think>17*23 is 391.</think> It is 391."))
        self.assertFalse(checks["reasoning"].ok)
        self.assertIn("INLINE", checks["reasoning"].detail)
        self.assertIn("client.py", checks["reasoning"].detail)

    def test_a_model_that_reasons_but_never_acts_fails_the_tool_check(self):
        checks = self._run(assistant_text("It is 391.", reasoning="Let me multiply."))
        self.assertTrue(checks["reasoning"].ok)
        self.assertFalse(checks["tool calling"].ok)
        self.assertIn("change nothing", checks["tool calling"].detail)

    def test_a_server_error_fails_both_rather_than_raising(self):
        """One failure must never hide the others — every check returns rather than raises."""
        broken = mock.Mock()
        broken.bind_tools.side_effect = ConnectionError("server went away")
        with mock.patch("agentfix.llm.client.make_chat_model", return_value=broken):
            checks = {c.name: c for c in doctor._check_reasoning_and_tools(LLMConfig())}
        self.assertFalse(checks["reasoning"].ok)
        self.assertFalse(checks["tool calling"].ok)
        self.assertIn("ConnectionError", checks["reasoning"].detail)


class TestReport(unittest.TestCase):
    def _report(self, checks):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = report(checks)
        return code, buffer.getvalue()

    def test_all_passing_is_ready(self):
        code, out = self._report(
            [Check("python", True, "3.12"), Check("generation", True, "40 tok/s")]
        )
        self.assertEqual(code, 0)
        self.assertIn("READY", out)
        self.assertIn("40 tok/s", out)

    def test_any_failure_is_not_ready_and_every_check_is_still_printed(self):
        code, out = self._report([Check("a", True, "ok"), Check("b", False, "broken")])
        self.assertEqual(code, 1)
        self.assertIn("NOT READY", out)
        self.assertIn("[PASS] a", out)
        self.assertIn("[FAIL] b", out)
