"""The trace: what makes a run debuggable rather than just pass/fail."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from agentfix.agent.trace import DETAIL_CLIP, Tracer, TraceEvent
from agentfix.llm.fake import assistant_text, assistant_tool_call


class TestTracer(unittest.TestCase):
    def test_events_are_collected_in_order(self):
        tracer = Tracer()
        tracer.record(TraceEvent(1, "llm", "assistant", "calls run_tests", 120, 0.4))
        tracer.record(TraceEvent(1, "tool", "run_tests", "Tests failed.", 120, 0.3))
        self.assertEqual([e.kind for e in tracer.events], ["llm", "tool"])

    def test_quiet_by_default(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            Tracer().record(TraceEvent(1, "llm", "assistant", "x", 1, 0.1))
        self.assertEqual(buffer.getvalue(), "")

    def test_verbose_prints_one_line_per_event(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            Tracer(verbose=True).record(
                TraceEvent(2, "tool", "run_tests", "Tests failed.", 300, 1.2)
            )
        printed = buffer.getvalue()
        self.assertIn("step 2", printed)
        self.assertIn("run_tests", printed)
        self.assertIn("300 tok", printed)
        self.assertEqual(len(printed.strip().splitlines()), 1)

    def test_newlines_are_flattened_and_long_detail_is_clipped(self):
        """One event, one line — a readable trace beats a faithful dump of file contents."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            Tracer(verbose=True).record(
                TraceEvent(1, "tool", "read_file", "a\nb\n" + "x" * 5000, 10, 0.0)
            )
        printed = buffer.getvalue()
        self.assertEqual(len(printed.strip().splitlines()), 1)
        self.assertLess(len(printed), DETAIL_CLIP + 200)

    def test_as_json_is_serialisable_for_the_eval_report(self):
        tracer = Tracer()
        tracer.record(TraceEvent(1, "llm", "assistant", "x", 120, 0.4))
        self.assertEqual(tracer.as_json()[0]["prompt_tokens"], 120)
        self.assertEqual(tracer.as_json()[0]["step"], 1)


class TestCallbackHooks(unittest.TestCase):
    """The hooks LangChain actually calls, driven directly — no graph, no model.

    These were previously only executed, never asserted on: a Tracer whose `step` never advanced,
    or whose `on_tool_error` did nothing, passed the entire suite.
    """

    def _llm_result(self, message):
        return LLMResult(generations=[[ChatGeneration(message=message)]])

    def test_each_model_turn_advances_the_step_and_records_its_context_size(self):
        tracer = Tracer()
        for turn, tokens in enumerate((99, 250), start=1):
            run_id = uuid4()
            tracer.on_chat_model_start({}, [[]], run_id=run_id)
            tracer.on_llm_end(
                self._llm_result(assistant_text("hi", prompt_tokens=tokens)), run_id=run_id
            )
            self.assertEqual(tracer.events[-1].step, turn)
            self.assertEqual(tracer.events[-1].prompt_tokens, tokens)

    def test_a_tool_event_carries_the_step_and_context_of_the_turn_that_asked_for_it(self):
        tracer = Tracer()
        llm_run = uuid4()
        tracer.on_chat_model_start({}, [[]], run_id=llm_run)
        tracer.on_llm_end(
            self._llm_result(assistant_tool_call("run_tests", {}, prompt_tokens=1234)),
            run_id=llm_run,
        )
        tool_run = uuid4()
        tracer.on_tool_start({"name": "run_tests"}, "{}", run_id=tool_run)
        tracer.on_tool_end(
            ToolMessage(content="All tests passed.", name="run_tests", tool_call_id="c1"),
            run_id=tool_run,
        )
        event = tracer.events[-1]
        self.assertEqual((event.kind, event.name), ("tool", "run_tests"))
        self.assertEqual(event.step, 1)
        self.assertEqual(event.prompt_tokens, 1234)

    def test_a_tool_that_raises_is_recorded_by_name(self):
        """`on_tool_error` is not told the name; it has to be remembered from on_tool_start."""
        tracer = Tracer()
        run_id = uuid4()
        tracer.on_tool_start({"name": "read_file"}, "{}", run_id=run_id)
        tracer.on_tool_error(RuntimeError("disk on fire"), run_id=run_id)
        event = tracer.events[-1]
        self.assertEqual(event.name, "read_file", "an unnamed error event is the useless one")
        self.assertIn("RuntimeError", event.detail)
        self.assertIn("disk on fire", event.detail)

    def test_a_reply_with_no_assistant_message_is_noted_rather_than_dropped(self):
        """A raise in a hook is swallowed by LangChain, so this must not raise — and a silent
        return would leave the turn invisible and the next tool line stamped with stale tokens.
        """
        tracer = Tracer()
        run_id = uuid4()
        tracer.on_chat_model_start({}, [[]], run_id=run_id)
        tracer.on_llm_end(
            self._llm_result(assistant_text("real", prompt_tokens=500)), run_id=run_id
        )

        second = uuid4()
        tracer.on_chat_model_start({}, [[]], run_id=second)
        tracer.on_llm_end(
            LLMResult(generations=[[Generation(text="not a chat reply")]]), run_id=second
        )
        self.assertEqual(tracer.events[-1].step, 2)
        self.assertIn("no assistant message", tracer.events[-1].detail)
        self.assertEqual(tracer.turn_prompt_tokens, 0, "a turn with no reply must not inherit 500")

    def test_an_empty_generation_list_does_not_raise(self):
        """`response.generations[0][0]` used to IndexError here, and LangChain would eat it."""
        tracer = Tracer()
        run_id = uuid4()
        tracer.on_chat_model_start({}, [[]], run_id=run_id)
        tracer.on_llm_end(LLMResult(generations=[]), run_id=run_id)
        self.assertIn("no assistant message", tracer.events[-1].detail)
