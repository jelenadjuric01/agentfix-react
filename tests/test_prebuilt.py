"""The prebuilt agent: what `create_agent` and its middleware do and do not give you.

Skipped unless the `prebuilt` extra is installed (`uv sync --extra prebuilt`). Everything here
runs the REAL prebuilt agent against the REAL tools in a real temp directory, with only the
model replaced — the same rule as tests/test_graph.py, so the two are comparable.
"""

from __future__ import annotations

import sys
import unittest

from agentfix.llm.fake import (
    FakeChatModel,
    assistant_invalid_tool_call,
    assistant_text,
    assistant_tool_call,
)
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool
from agentfix.tools.tests_tool import RunTestsTool
from tests.support import TempDirTestCase

try:  # the extra is optional; the rest of the suite must not need it
    from agentfix.agent.prebuilt import build_prebuilt_agent, prebuilt_solved

    PREBUILT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on which extras are installed
    PREBUILT_AVAILABLE = False

BUGGY = "def total(prices):\n    return sum(prices) - 1\n"
FIXED = "def total(prices):\n    return sum(prices)\n"
SUITE = (
    "import unittest\n\n"
    "from cart import total\n\n\n"
    "class TestCart(unittest.TestCase):\n"
    "    def test_total(self):\n"
    "        self.assertEqual(total([1, 2]), 3)\n"
)


@unittest.skipUnless(PREBUILT_AVAILABLE, "needs the prebuilt extra: uv sync --extra prebuilt")
class PrebuiltTestCase(TempDirTestCase):
    """The same red one-file project tests/test_graph.py uses."""

    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "cart.py").write_text(BUGGY, encoding="utf-8")
        (self.tmp / "test_cart.py").write_text(SUITE, encoding="utf-8")
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

    def run_with(self, replies, max_steps=None):
        replies = list(replies)
        llm = FakeChatModel(replies=replies)
        app = build_prebuilt_agent(
            llm, self.tools, max_steps=len(replies) if max_steps is None else max_steps
        )
        final = app.invoke({"messages": [("user", "The tests fail. Fix the bug.")]})
        return final, llm


class TestWhatTheFrameworkGives(PrebuiltTestCase):
    def test_it_solves_the_task(self):
        """The tool-calling loop itself needs none of graph.py."""
        final, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("read_file", {"path": "cart.py"}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed the off-by-one."),
            ]
        )
        self.assertTrue(prebuilt_solved(final))
        self.assertEqual((self.tmp / "cart.py").read_text(), FIXED)

    def test_the_step_budget_is_model_calls_not_node_executions(self):
        """ModelCallLimitMiddleware counts the same thing AgentState.step counts."""
        _, llm = self.run_with(
            [assistant_tool_call("list_files", {}, call_id=f"c{i}") for i in range(10)],
            max_steps=3,
        )
        self.assertEqual(llm.index, 3, "the run must stop after three model calls")

    def test_prose_while_the_suite_is_red_does_not_end_the_run(self):
        """after_model + jump_to="model" is what makes the verified stop expressible."""
        _, llm = self.run_with(
            [assistant_text("I am confident it is fine."), assistant_text("still fine")],
            max_steps=2,
        )
        self.assertEqual(llm.index, 2, "the model was sent back rather than believed")
        self.assertTrue(
            any("tests have not passed" in str(m.content) for m in llm.calls[-1]),
            "the nudge reached the model",
        )

    def test_an_identical_repeated_call_is_not_executed_again(self):
        """wrap_tool_call can answer a call without running it — that is the whole guard."""
        final, _ = self.run_with(
            [
                assistant_tool_call("list_files", {}, call_id="c1"),
                assistant_tool_call("list_files", {}, call_id="c2"),
                assistant_text("done"),
            ],
            max_steps=3,
        )
        answers = [str(m.content) for m in final["messages"] if getattr(m, "tool_call_id", None)]
        self.assertTrue(any("already called" in a for a in answers))


class TestMiddlewareOrderIsLoadBearing(PrebuiltTestCase):
    def test_the_step_budget_only_applies_if_the_limit_is_ordered_last(self):
        """The finding that justifies this file existing.

        `after_model` hooks run in reverse list order and routing obeys whichever `jump_to`
        reached the state last, so a budget middleware placed before one that jumps back to the
        model is silently overruled. Nothing warns you; the agent runs until something else
        stops it — here, the scripted fake running out of replies.
        """
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelCallLimitMiddleware

        from agentfix.agent.graph import system_prompt
        from agentfix.agent.prebuilt import LoopGuard, VerifiedStop

        def calls_made(limit_first: bool) -> int:
            llm = FakeChatModel(replies=[assistant_text(f"no {i}") for i in range(12)])
            limit = ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")
            order = [limit, LoopGuard(), VerifiedStop()]
            if not limit_first:
                order = [LoopGuard(), VerifiedStop(), limit]
            app = create_agent(
                model=llm,
                tools=self.tools,
                system_prompt=system_prompt(self.tools),
                middleware=order,
            )
            try:
                app.invoke({"messages": [("user", "fix")]})
            except AssertionError:
                pass  # the fake exhausted its script — the symptom of an ignored budget
            return llm.index

        self.assertEqual(calls_made(limit_first=False), 3, "limit last: budget respected")
        self.assertGreater(calls_made(limit_first=True), 3, "limit first: budget ignored")


class TestWhatIsStillMissing(PrebuiltTestCase):
    """The gaps that keep agent/graph.py from collapsing into a constructor call."""

    def test_arguments_that_are_not_valid_json_go_unanswered(self):
        """Still dropped, exactly as ToolNode drops them. Our graph answers them by hand.

        The consequence is not cosmetic: the API rejects any request that leaves a
        `tool_call_id` unanswered, so against a real server the NEXT turn fails.
        """
        final, _ = self.run_with(
            [
                assistant_invalid_tool_call("read_file", '{"path": "cart.py"'),
                assistant_text("giving up"),
            ],
            max_steps=2,
        )
        answers = [m for m in final["messages"] if getattr(m, "tool_call_id", None)]
        self.assertEqual(answers, [], "if this ever fails, the framework fixed the gap")

    def test_the_guards_counters_leak_between_runs_of_one_agent(self):
        """The consequence of keeping guard state on the middleware instead of in the state.

        Replaces an earlier assertion that `guard_hits` is absent from the state, which could
        only ever fail if the framework adopted our key names. This asserts the behaviour that
        absence actually causes.
        """
        llm = FakeChatModel(
            replies=[
                assistant_tool_call("list_files", {}, call_id="a1"),
                assistant_text("done"),
                assistant_tool_call("list_files", {}, call_id="b1"),
                assistant_text("done again"),
            ]
        )
        app = build_prebuilt_agent(llm, self.tools, max_steps=2)
        app.invoke({"messages": [("user", "task one")]})
        second = app.invoke({"messages": [("user", "a brand new task")]})

        answers = [str(m.content) for m in second["messages"] if getattr(m, "tool_call_id", None)]
        self.assertTrue(
            any("already called" in a for a in answers),
            "run 2's opening call is refused as a repeat of run 1's last one",
        )

    def test_a_stuck_model_runs_to_the_budget_instead_of_being_abandoned(self):
        """The guard can answer a repeated call but cannot end the run.

        agent/graph.py abandons after MAX_GUARD_HITS repeats; here the model keeps its whole
        budget, which is the difference between a seam and a policy.
        """
        _, llm = self.run_with(
            [assistant_tool_call("list_files", {}, call_id=f"c{i}") for i in range(9)],
            max_steps=9,
        )
        self.assertEqual(llm.index, 9, "nine identical calls cost nine model turns")

    def test_a_checkpointer_is_refused_rather_than_silently_mis_reporting(self):
        """The verdict is recomputed from artifacts a checkpoint round-trip turns into dicts.

        Measured before this guard existed: a live run reported solved, the same thread resumed
        reported unsolved, and VerifiedStop then nudged a green suite until the budget ran out.
        Refusing names the limitation where someone would hit it.
        """
        from langgraph.checkpoint.memory import InMemorySaver

        with self.assertRaises(NotImplementedError) as ctx:
            build_prebuilt_agent(
                FakeChatModel(replies=[]), self.tools, checkpointer=InMemorySaver()
            )
        self.assertIn("nowhere to keep the verdict", str(ctx.exception))

    def test_the_verdict_has_to_be_recomputed_from_the_messages(self):
        """`create_agent` carries its own state, so there is no tests_passed to read.

        The `assertNotIn` half is weak on its own — it only fails if the framework adopts our
        key name. The half that carries weight is that `prebuilt_solved` has to re-fold the
        whole history to answer a question `agent/graph.py` reads from one bool.
        """
        final, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertNotIn("tests_passed", final)
        self.assertTrue(prebuilt_solved(final))
        # And the artifacts it depends on are live objects, not the dicts a checkpoint returns.
        artifacts = [
            m.artifact for m in final["messages"] if getattr(m, "artifact", None) is not None
        ]
        self.assertTrue(artifacts)
        self.assertFalse([a for a in artifacts if isinstance(a, dict)])
