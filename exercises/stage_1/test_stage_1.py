"""Stage 1 — a turn that changed nothing.

Every test here drives the REAL graph against the REAL tools in a real temp directory. Only the
model is scripted, so none of this needs Ollama running — and the scripted turns are shaped like
the ones Mellum2-Thinking actually produces, with the reasoning in the field the real client puts
it in. See llm/fake.py.

`assistant_thinking(...)` is the turn this whole stage is about: reasoning on its own channel,
empty content, and no tool call. The characteristic failure of a thinking model, on demand.
"""

from __future__ import annotations

import sys
import unittest

from agentgraph.agent.graph import (
    MAX_IDLE_TURNS,
    NUDGE,
    NUDGE_AFTER_THINKING,
    acted,
    run_agent,
)
from agentgraph.agent.trace import Tracer
from agentgraph.llm.fake import (
    FakeChatModel,
    assistant_text,
    assistant_thinking,
    assistant_tool_call,
)
from agentgraph.sandbox.subprocess_backend import SubprocessBackend
from agentgraph.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentgraph.tools.tests_tool import RunTestsTool
from tests.support import TempDirTestCase, make_task

BUGGY = "def total(prices):\n    return sum(prices) - 1\n"
FIXED = "def total(prices):\n    return sum(prices)\n"
SUITE = (
    "import unittest\n\n"
    "from cart import total\n\n\n"
    "class TestCart(unittest.TestCase):\n"
    "    def test_total(self):\n"
    "        self.assertEqual(total([1, 2]), 3)\n"
)

THOUGHT = "The test expects 3 and got 2, so the subtraction in total() is wrong."


class Stage1TestCase(TempDirTestCase):
    """A real one-file project that starts red, plus the four tools bound to it."""

    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "cart.py").write_text(BUGGY, encoding="utf-8")
        (self.tmp / "test_cart.py").write_text(SUITE, encoding="utf-8")
        self.task = make_task(self.tmp)
        self.tools = [
            ListFilesTool(root=self.tmp),
            ReadFileTool(root=self.tmp),
            WriteFileTool(root=self.tmp),
            RunTestsTool(
                root=self.tmp,
                command=(sys.executable, "-m", "unittest", "discover", "-q"),
                backend=SubprocessBackend(),
                timeout_s=30,
            ),
        ]

    def run_with(self, replies, max_steps=None, tracer=None):
        # The script is the budget by default. A reply that acted on nothing is nudged rather
        # than accepted, and the nudge costs a turn — so a script of N replies needs N steps, or
        # the fake runs off the end of its own screenplay.
        replies = list(replies)
        llm = FakeChatModel(replies=replies)
        result = run_agent(
            self.task,
            llm,
            self.tools,
            max_steps=len(replies) if max_steps is None else max_steps,
            tracer=tracer,
        )
        return result, llm

    def nudges_sent(self, llm):
        """Every correction the graph injected, read off the last history the model was sent."""
        from langchain_core.messages import HumanMessage

        return [message.content for message in llm.calls[-1] if isinstance(message, HumanMessage)]


class TestWhatCountsAsActing(unittest.TestCase):
    """Blank 1. Two lines of test for one line of code, because everything else rests on it."""

    def test_a_turn_with_a_tool_call_acted(self):
        self.assertTrue(acted(assistant_tool_call("run_tests", {})))

    def test_a_turn_that_only_reasoned_did_not_act(self):
        """Three hundred tokens of deliberation is not an action."""
        self.assertFalse(acted(assistant_thinking(THOUGHT)))

    def test_prose_is_not_an_action_either(self):
        self.assertFalse(acted(assistant_text("I have fixed the bug.")))

    def test_reasoning_does_not_stop_a_tool_call_from_counting(self):
        """ReAct means thinking and acting in the SAME turn. That turn acted."""
        self.assertTrue(acted(assistant_tool_call("run_tests", {}, reasoning=THOUGHT)))


class TestTheThinkingGuardEndsTheRun(Stage1TestCase):
    def test_a_model_that_only_thinks_is_abandoned_rather_than_nudged_forever(self):
        # A generous step budget, so that what stops this run is the guard and not the budget.
        # Without the guard the script would be nudged around the loop until the budget ran out,
        # spending the most expensive kind of turn there is each time round.
        result, llm = self.run_with(
            [assistant_thinking(f"{THOUGHT} Let me consider it further. #{n}") for n in range(9)],
            max_steps=9,
        )
        self.assertFalse(result.solved)
        # The literal 2, deliberately, not MAX_IDLE_TURNS. Asserting against the constant the
        # code already uses is a tautology — it would pass at any value, so it pins nothing.
        self.assertEqual(result.steps_used, 2, "two wasted turns and the run stops")
        self.assertLess(llm.index, 9, "the run stopped early instead of burning the budget")

    def test_the_thinking_budget_is_two_turns(self):
        """The value itself, stated once, so a change to it cannot pass silently."""
        self.assertEqual(MAX_IDLE_TURNS, 2)

    def test_a_silent_turn_counts_the_same_as_a_thinking_one(self):
        """`idle_turns` is about what the turn MOVED, not about whether it deliberated.

        A reply cut off by `max_tokens` before it reached its tool call reasoned and asked for
        nothing; a model that skipped thinking and said "looks fine" did neither. Both changed
        nothing, and both cost the same.
        """
        result, _ = self.run_with(
            [assistant_text(f"Looks fine to me. #{n}") for n in range(9)], max_steps=9
        )
        self.assertFalse(result.solved)
        self.assertEqual(result.steps_used, 2)

    def test_the_trace_says_why_the_run_was_abandoned(self):
        """A run that ends on a guard and a run that ends on the budget are different failures."""
        tracer = Tracer()
        self.run_with(
            [assistant_thinking(f"thinking {n}") for n in range(4)], max_steps=4, tracer=tracer
        )
        notes = [event.detail for event in tracer.events]
        self.assertTrue(
            any("no tool call" in note for note in notes),
            f"the abandonment should be recorded, got {notes}",
        )


class TestThinkingIsStillAllowed(Stage1TestCase):
    """The other half of the decision, and the easier half to get wrong."""

    def test_one_thinking_turn_is_allowed_because_planning_is_not_stalling(self):
        """MAX_IDLE_TURNS is 2, not 1 — a turn spent planning must not end the run."""
        result, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_thinking("The failure is in total(). Planning the fix."),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed the off-by-one."),
            ],
            max_steps=6,
        )
        self.assertTrue(result.solved, "one thinking turn mid-run is legitimate")
        self.assertIn(NUDGE_AFTER_THINKING, self.nudges_sent(llm))

    def test_acting_resets_the_idle_count(self):
        """Blank 2, and the reason it cannot be a reducer.

        Two thinking turns that are not consecutive are not a stall. Without the reset they
        accumulate, and this run is abandoned on its fourth turn with a fix it never wrote.
        """
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
        self.assertEqual((self.tmp / "cart.py").read_text(), FIXED)


class TestTheTestsStillDecide(Stage1TestCase):
    """Ordering. The guard must not be reached on a turn that had nothing left to do."""

    def test_a_run_still_ends_successfully_when_the_tests_pass(self):
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed the off-by-one."),
            ]
        )
        self.assertTrue(result.solved)

    def test_a_thinking_turn_on_a_green_suite_ends_the_run_rather_than_being_nudged(self):
        """The closing turn of a solved run is a turn that asked for nothing.

        Here it is a turn that reasoned about the fix and stopped, which is exactly what this
        model does when the work is done. `is_done` has to be read before anything else in the
        tail — a tail that nudges first asks for a fifth turn that was never scripted, and one
        that guards first would report a solved run as a stalled model.
        """
        result, llm = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_thinking("The suite is green, so the rounding fix was the whole bug."),
            ]
        )
        self.assertTrue(result.solved)
        self.assertEqual(llm.index, 3, "the graph took exactly the scripted turns")

    def test_a_model_that_merely_claims_success_is_not_believed(self):
        """The tests were never run, so nothing is green — whatever the model reasoned its way to."""
        result, _ = self.run_with(
            [
                assistant_text("I have fixed the bug.", reasoning="The subtraction was wrong."),
                assistant_text("Really, it is fixed."),
            ],
            max_steps=2,
        )
        self.assertFalse(result.solved, "prose is not evidence, and neither is reasoning")

    def test_the_step_budget_still_ends_the_run(self):
        """No nudging past the budget, or a stubborn model never stops.

        The tool call on every turn keeps `idle_turns` at zero, so the thinking guard never
        fires here and the budget is the only thing left to stop the run.
        """
        result, llm = self.run_with(
            [assistant_tool_call("list_files", {}) for _ in range(10)], max_steps=3
        )
        self.assertFalse(result.solved)
        self.assertEqual(llm.index, 3, "the run must not exceed its step budget")


class TestTheNudgeMatchesTheFailure(Stage1TestCase):
    """Blank 4. Two nudges, because the two failures deserve different corrections."""

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
        self.assertIn(NUDGE_AFTER_THINKING, self.nudges_sent(llm))

    def test_a_model_that_said_nothing_useful_gets_the_plain_nudge(self):
        """Nothing to correct about the reasoning of a turn that did not reason."""
        _, llm = self.run_with(
            [
                assistant_text("I think it is fine."),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed it."),
            ],
            max_steps=6,
        )
        self.assertIn(NUDGE, self.nudges_sent(llm))


if __name__ == "__main__":
    unittest.main()
