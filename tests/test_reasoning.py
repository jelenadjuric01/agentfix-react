"""Reasoning: the behaviour that makes this the ReAct edition rather than the last one.

Everything here runs offline. `FakeChatModel` puts reasoning in exactly the field the real
client puts it in — `additional_kwargs["reasoning_content"]`, see llm/fake.py — so these tests
exercise the real graph, the real router and the real tracer against turns shaped like the ones
Mellum2-Thinking actually produces.

Four things are pinned, and each one is a bug this edition could have shipped:

  1. reasoning is READ from the right place, and counted
  2. the trace REPORTS it — the previous edition would have said "(NO REASONING)" here
  3. a model that thinks and never acts is CAUGHT, and told the right thing
  4. the action guard IGNORES reasoning, so a repeat dressed in new words is still a repeat
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from agentgraph.agent.graph import (
    MAX_IDLE_TURNS,
    NUDGE,
    NUDGE_AFTER_THINKING,
    UNREADABLE_REPLY,
    acted,
    call_signature,
)
from agentgraph.agent.graph import run_agent
from agentgraph.agent.trace import TraceEvent, Tracer, describe, reasoning_of
from agentgraph.llm.fake import (
    assistant_text,
    assistant_thinking,
    assistant_tool_call,
    unreadable_reply,
)
from agentgraph.llm.fake import FakeChatModel
from tests.test_graph import FIXED, GraphTestCase

THOUGHT = "The test expects 3 and got 2, so the subtraction in total() is wrong."


class TestReasoningIsReadFromTheRightPlace(unittest.TestCase):
    def test_reasoning_comes_off_additional_kwargs(self):
        message = assistant_tool_call("run_tests", {}, reasoning=THOUGHT)
        self.assertEqual(reasoning_of(message), THOUGHT)

    def test_a_turn_without_reasoning_reads_as_empty_not_as_none(self):
        """`reasoning_of` always returns a string, so every caller can treat it as falsy."""
        self.assertEqual(reasoning_of(assistant_tool_call("run_tests", {})), "")

    def test_reasoning_does_not_leak_into_the_answer(self):
        """`content` is the answer only. A thinking turn that acts usually has no answer at all.

        This is the property `reasoning=True` buys: with it unset the same text would arrive
        inline in `content` wrapped in `<think>` tags, and `write_file` would be handed a
        "complete file" with a monologue at the top of it.
        """
        message = assistant_tool_call("run_tests", {}, reasoning=THOUGHT)
        self.assertEqual(message.text, "")

    def test_a_thinking_turn_asked_for_nothing(self):
        self.assertFalse(acted(assistant_thinking(THOUGHT)))

    def test_a_turn_that_reasoned_and_called_a_tool_counts_as_acting(self):
        self.assertTrue(acted(assistant_tool_call("run_tests", {}, reasoning=THOUGHT)))


class TestTheTraceReportsReasoning(unittest.TestCase):
    """The regression the port could most easily have shipped, and shipped silently."""

    def test_describe_does_not_claim_no_reasoning_when_the_model_reasoned(self):
        message = assistant_tool_call("run_tests", {}, reasoning=THOUGHT)
        summary = describe(message)
        self.assertIn("calls run_tests", summary)
        self.assertNotIn("NO REASONING", summary)

    def test_describe_still_flags_a_turn_that_acted_without_thinking(self):
        """The marker survives, but now it means what it says."""
        self.assertIn("NO REASONING", describe(assistant_tool_call("run_tests", {})))

    def test_verbose_output_prints_the_thinking_under_the_turn(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            Tracer(verbose=True).record(
                TraceEvent(1, "llm", "assistant", "calls run_tests", 120, 0.4, THOUGHT)
            )
        printed = buffer.getvalue()
        self.assertIn("thinks", printed)
        self.assertIn("subtraction", printed)
        self.assertEqual(
            len(printed.strip().splitlines()), 2, "the action line, then the thinking under it"
        )

    def test_reasoning_survives_into_the_eval_json(self):
        tracer = Tracer()
        tracer.record(TraceEvent(1, "llm", "assistant", "calls run_tests", 10, 0.1, THOUGHT))
        self.assertEqual(tracer.as_json()[0]["reasoning"], THOUGHT)


class TestReasoningIsCounted(GraphTestCase):
    def test_only_the_turns_that_reasoned_are_counted(self):
        result, _ = self.run_with(
            [
                assistant_tool_call("run_tests", {}, reasoning=THOUGHT),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}, reasoning="Confirming the fix."),
                assistant_text("Fixed the off-by-one.", reasoning="It passes now."),
            ]
        )
        self.assertTrue(result.solved)
        self.assertEqual(result.steps_used, 4)
        self.assertEqual(result.reasoning_turns, 3)

    def test_an_agent_that_never_reasons_reports_zero(self):
        """The previous edition's measured result, reproducible here on demand."""
        result, _ = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertTrue(result.solved)
        self.assertEqual(result.reasoning_turns, 0)


class TestTheThinkingLoopGuard(GraphTestCase):
    """A model that deliberates instead of working. The characteristic failure here."""

    def test_a_model_that_only_thinks_is_abandoned_rather_than_nudged_forever(self):
        # A generous step budget, so that what stops the run is the thinking guard and not the
        # budget. Without the guard this script would be nudged around the loop until the
        # budget ran out, spending the most expensive kind of turn each time.
        result, llm = self.run_with(
            [assistant_thinking(f"{THOUGHT} Let me consider it further. #{n}") for n in range(9)],
            max_steps=9,
        )
        self.assertFalse(result.solved)
        # The literal 2, deliberately, not MAX_IDLE_TURNS. Asserting against the constant the
        # code already uses is a tautology: it passes at any value, so it pins nothing. This is
        # the contract — two wasted turns and the run stops — and changing the constant should
        # have to be a deliberate edit to this line.
        self.assertEqual(result.steps_used, 2)
        self.assertLess(llm.index, 9, "the run stopped early instead of burning the budget")

    def test_the_thinking_budget_is_two_turns(self):
        """The value itself, stated once, so a change to it cannot pass silently."""
        self.assertEqual(MAX_IDLE_TURNS, 2)

    def test_one_thinking_turn_is_allowed_because_planning_is_not_stalling(self):
        """MAX_IDLE_TURNS is 2, not 1 — a turn spent planning must not end the run."""
        result, _ = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_thinking("The failure is in total(). Planning the fix."),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed it."),
            ],
            max_steps=6,
        )
        self.assertTrue(result.solved, "one thinking turn mid-run is legitimate")

    def test_acting_resets_the_idle_count(self):
        """Otherwise thinking turns spread across a long run would accumulate into a false stall."""
        result, _ = self.run_with(
            [
                assistant_thinking("Let me plan."),
                assistant_tool_call("run_tests", {}),
                assistant_thinking("Now let me think again."),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed it."),
            ],
            max_steps=8,
        )
        self.assertTrue(result.solved)

    def test_the_trace_says_why_the_run_was_abandoned(self):
        tracer = Tracer()
        self.run_with(
            [assistant_thinking(f"thinking {n}") for n in range(4)], max_steps=4, tracer=tracer
        )
        notes = [event.detail for event in tracer.events]
        self.assertTrue(
            any("no tool call" in note for note in notes),
            f"the abandonment should be recorded, got {notes}",
        )


class TestTheNudgeMatchesTheFailure(GraphTestCase):
    """Two nudges, because "said nothing" and "reasoned and stopped" need different corrections."""

    def _human_texts(self, llm):
        """Every nudge the graph injected, read off the last history the model was sent."""
        from langchain_core.messages import HumanMessage

        return [message.content for message in llm.calls[-1] if isinstance(message, HumanMessage)]

    def test_a_model_that_reasoned_and_did_not_act_is_told_reasoning_is_not_an_action(self):
        _, llm = self.run_with(
            [
                assistant_thinking(THOUGHT),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed it."),
            ],
            max_steps=6,
        )
        self.assertIn(NUDGE_AFTER_THINKING, self._human_texts(llm))

    def test_a_model_that_said_nothing_useful_gets_the_plain_nudge(self):
        _, llm = self.run_with(
            [
                assistant_text("I think it is fine."),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed it."),
            ],
            max_steps=6,
        )
        self.assertIn(NUDGE, self._human_texts(llm))


class TestTheActionGuardIgnoresReasoning(GraphTestCase):
    """Fresh reasoning must not buy a repeated call another turn."""

    def test_the_signature_is_the_same_whatever_the_model_was_thinking(self):
        call = {"name": "read_file", "args": {"path": "cart.py"}, "id": "c1"}
        self.assertEqual(call_signature(call), call_signature(dict(call)))

    def test_the_same_call_with_different_reasoning_is_still_guarded(self):
        """The hole a reasoning-aware signature would have opened.

        A small model rarely repeats itself word for word — it talks itself into the same dead
        end by a slightly different route. Include the reasoning in the signature and every
        repeat looks novel, the guard never fires, and the run burns its budget re-reading one
        file.
        """
        tracer = Tracer()
        result, _ = self.run_with(
            [
                assistant_tool_call("read_file", {"path": "cart.py"}, reasoning="Let me look."),
                assistant_tool_call(
                    "read_file", {"path": "cart.py"}, reasoning="On reflection, look again."
                ),
                assistant_tool_call(
                    "read_file", {"path": "cart.py"}, reasoning="A third, quite different, look."
                ),
                assistant_text("done"),
            ],
            max_steps=4,
            tracer=tracer,
        )
        self.assertFalse(result.solved)
        guarded = [event for event in tracer.events if "guarded" in event.detail]
        self.assertTrue(guarded, "the repeat should have been refused despite new reasoning")


class TestAnUnreadableReplyIsRecoverable(GraphTestCase):
    """A reply the client cannot parse must cost a turn, not the run.

    `ChatOllama` raises `OutputParserException` on malformed tool-call arguments rather than
    reporting them as a call it could not read, so this is the only bad-JSON path there is. Before the recovery it propagated out of the agent and the task was
    recorded as a CRASH: a harness failure, for what is really just the model getting one reply
    wrong.
    """

    def test_the_run_survives_and_still_solves(self):
        result, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}, reasoning="Measure first."),
                unreadable_reply(),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed the off-by-one."),
            ],
            max_steps=8,
        )
        self.assertTrue(result.solved, "one unreadable reply must not end the run")
        self.assertEqual(llm.index, 5, "every scripted turn was used, including the bad one")

    def test_the_bad_turn_still_costs_a_step(self):
        """It consumed a model call, so it is on the bill. Otherwise the budget is a lie."""
        result, _ = self.run_with(
            [
                unreadable_reply(),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=6,
        )
        self.assertTrue(result.solved)
        self.assertEqual(result.steps_used, 4)

    def test_the_model_is_told_what_went_wrong(self):
        """A retry with no explanation is just the same reply again."""
        from langchain_core.messages import HumanMessage

        _, llm = self.run_with(
            [
                unreadable_reply(),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=6,
        )
        sent = [m.content for m in llm.calls[-1] if isinstance(m, HumanMessage)]
        self.assertIn(UNREADABLE_REPLY, sent)

    def test_no_assistant_turn_is_fabricated_for_a_reply_that_never_parsed(self):
        """There was no message. Inventing an empty one would put words in the model's mouth."""
        from langchain_core.messages import AIMessage

        _, llm = self.run_with(
            [
                unreadable_reply(),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=6,
        )
        # The history the model saw on its second turn: one system, one task, one correction.
        second_turn = llm.calls[1]
        self.assertEqual(
            [m for m in second_turn if isinstance(m, AIMessage)],
            [],
            "the unreadable turn must leave no assistant message behind",
        )

    def test_a_model_that_never_emits_readable_json_is_abandoned(self):
        """Bounded by the same guard as thinking: two turns that changed nothing, and stop."""
        result, llm = self.run_with([unreadable_reply() for _ in range(9)], max_steps=9)
        self.assertFalse(result.solved)
        self.assertEqual(result.steps_used, 2)
        self.assertLess(llm.index, 9, "it gave up instead of burning the whole budget")

    def test_an_unreadable_reply_on_the_last_allowed_turn_ends_the_run(self):
        """The step budget still applies to a turn that could not be read.

        Otherwise the retry would be free: a reply that failed to parse on the final permitted
        turn would be sent back to the model, buying a turn the budget had already spent.
        """
        result, llm = self.run_with([unreadable_reply()], max_steps=1)
        self.assertFalse(result.solved)
        self.assertEqual(result.steps_used, 1)
        self.assertEqual(llm.index, 1, "no extra turn was granted by the retry path")

    def test_a_readable_turn_resets_the_allowance(self):
        """Otherwise two bad replies far apart in a long run would end it."""
        result, _ = self.run_with(
            [
                unreadable_reply(),
                assistant_tool_call("run_tests", {}),
                unreadable_reply(),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=8,
        )
        self.assertTrue(result.solved)

    def test_the_trace_records_it(self):
        tracer = Tracer()
        self.run_with([unreadable_reply(), unreadable_reply()], max_steps=4, tracer=tracer)
        details = [event.detail for event in tracer.events]
        self.assertTrue(
            any("unreadable reply" in d for d in details),
            f"the failure should be visible in the trace, got {details}",
        )

    def test_a_connection_error_is_not_swallowed(self):
        """Only a parse failure is recoverable. A dead server must still stop the run.

        The recovery catches `OutputParserException` specifically and not `Exception`, because
        "the model said something odd" and "there is no model" need opposite responses.
        """
        llm = FakeChatModel(replies=[ConnectionError("server went away")])
        with self.assertRaises(ConnectionError):
            run_agent(self.task, llm, self.tools, max_steps=4)
