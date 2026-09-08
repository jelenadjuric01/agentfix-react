"""The agent graph: the step budget, the verification stop condition, and the loop guard.

These tests run the REAL graph against REAL tools in a REAL temp directory. Only the model is
replaced. Nothing is patched — that is what makes them meaningful: the suite really is red
before the agent's write and really is green after it.
"""

from __future__ import annotations

import sys
import time
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from agentgraph.agent.graph import (
    MAX_GUARD_HITS,
    NUDGE,
    build_graph,
    run_agent,
    system_prompt,
)
from agentgraph.agent.state import initial_state
from agentgraph.agent.trace import Tracer
from agentgraph.llm.fake import (
    FakeChatModel,
    assistant_text,
    assistant_tool_call,
    assistant_tool_calls,
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


class GraphTestCase(TempDirTestCase):
    """A real one-file project that starts red, plus the tools bound to it."""

    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "cart.py").write_text(BUGGY, encoding="utf-8")
        (self.tmp / "test_cart.py").write_text(SUITE, encoding="utf-8")
        self.task = make_task(self.tmp)
        self.run_tests = RunTestsTool(
            root=self.tmp,
            command=(sys.executable, "-m", "unittest", "discover", "-q"),
            backend=SubprocessBackend(),
            timeout_s=30,
        )
        self.tools = [
            ListFilesTool(root=self.tmp),
            ReadFileTool(root=self.tmp),
            WriteFileTool(root=self.tmp),
            self.run_tests,
        ]

    def run_with(self, replies, max_steps=None, tracer=None):
        # The script is the budget by default. A prose reply while the tests are still red is
        # nudged rather than accepted, and the nudge costs a turn — so a script of N replies
        # needs exactly N steps, or the fake runs off the end of its own screenplay.
        replies = list(replies)
        max_steps = len(replies) if max_steps is None else max_steps
        llm = FakeChatModel(replies=replies)
        result = run_agent(self.task, llm, self.tools, max_steps=max_steps, tracer=tracer)
        return result, llm


class TestConstants(GraphTestCase):
    """The budgets themselves, stated as literals once so a change cannot pass silently."""

    def test_the_repeat_budget_is_three_strikes(self):
        self.assertEqual(MAX_GUARD_HITS, 3)


class TestSystemPrompt(GraphTestCase):
    def test_the_tool_names_come_from_the_registered_tools(self):
        """Derived, not hardcoded, so the prompt and the schemas cannot drift apart."""
        prompt = system_prompt(self.tools)
        for name in ("list_files", "read_file", "write_file", "run_tests"):
            self.assertIn(name, prompt)


class TestHappyPath(GraphTestCase):
    def test_the_agent_solves_the_task_and_reports_what_it_cost(self):
        result, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("read_file", {"path": "cart.py"}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed the off-by-one."),
            ]
        )
        self.assertTrue(result.solved)
        self.assertEqual(result.steps_used, 5)
        self.assertEqual(llm.index, 5, "the graph took exactly the scripted turns")
        self.assertEqual((self.tmp / "cart.py").read_text(), FIXED)
        self.assertGreater(result.prompt_tokens, 0)
        self.assertGreater(result.peak_prompt_tokens, 0)

    def test_the_token_accounting_sums_and_the_peak_is_a_high_water_mark(self):
        """Pins all three numeric reducers at once.

        Without this, dropping `operator.add` from the token counters or `keep_larger` from the
        peak changed nothing that any test noticed — the only assertions were `> 0`.
        """
        result, _ = self.run_with(
            [
                assistant_tool_call("run_tests", {}, prompt_tokens=100),
                assistant_tool_call("list_files", {}, prompt_tokens=300),
                assistant_text("done", prompt_tokens=200),
            ]
        )
        self.assertEqual(result.prompt_tokens, 600, "the bill is the sum of every turn")
        self.assertEqual(result.peak_prompt_tokens, 300, "the peak is the largest single prompt")
        self.assertGreater(result.completion_tokens, 0)

    def test_the_history_is_append_only(self):
        """Byte-stable prefix -> the server's KV cache stays valid across turns."""
        _, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        for earlier, later in zip(llm.calls, llm.calls[1:]):
            self.assertEqual(later[: len(earlier)], earlier)

    def test_the_tool_call_id_is_carried_back(self):
        """The API pairs each answer to its question by id; skip one and the next call fails."""
        _, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}, call_id="abc123"),
                assistant_text("done"),
            ]
        )
        answers = [m for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        self.assertEqual([m.tool_call_id for m in answers], ["abc123"])

    def test_several_calls_in_one_turn_are_all_answered(self):
        """A turn answering only the first call would otherwise pass every test."""
        _, llm = self.run_with(
            [
                assistant_tool_calls(
                    [("run_tests", {}), ("list_files", {})], call_ids=("c1", "c2")
                ),
                assistant_text("done"),
            ]
        )
        ids = [m.tool_call_id for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        self.assertEqual(ids, ["c1", "c2"])


class TestFailuresBecomeObservations(GraphTestCase):
    """Every way a model can get a tool call wrong must be recoverable, not fatal."""

    def _last_tool_content(self, llm) -> str:
        tools = [m for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        return str(tools[-1].content)

    def test_a_hallucinated_tool_name_does_not_end_the_run(self):
        result, llm = self.run_with(
            [
                assistant_tool_call("frobnicate", {"x": 1}),
                assistant_text("giving up"),
            ]
        )
        self.assertFalse(result.solved)
        self.assertIn("frobnicate", self._last_tool_content(llm))

    def test_a_missing_required_argument_does_not_end_the_run(self):
        _, llm = self.run_with(
            [
                assistant_tool_call("read_file", {}),
                assistant_text("giving up"),
            ]
        )
        content = self._last_tool_content(llm).lower()
        self.assertTrue("path" in content or "required" in content)

    def test_a_tool_that_raises_does_not_end_the_run(self):
        """ToolNode's default lets the exception through; the graph opts back in."""
        broken = ReadFileTool(root=self.tmp)

        def explode(**_kwargs):
            raise RuntimeError("disk on fire")

        object.__setattr__(broken, "_run", explode)
        self.tools[1] = broken
        result, llm = self.run_with(
            [
                assistant_tool_call("read_file", {"path": "cart.py"}),
                assistant_text("giving up"),
            ]
        )
        self.assertFalse(result.solved)
        self.assertTrue(self._last_tool_content(llm))


class TestStopCondition(GraphTestCase):
    def test_not_done_before_the_tests_have_ever_run(self):
        """The verdict starts False, so an agent that never runs the tests never solves."""
        result, _ = self.run_with([assistant_text("looks fine to me")], max_steps=1)
        self.assertFalse(result.solved)

    def test_not_done_when_the_model_only_claims_success(self):
        """The whole point: a model announcing victory is not evidence."""
        result, _ = self.run_with(
            [
                assistant_text("I have fixed the bug."),
                assistant_text("Really, it is fixed."),
            ],
            max_steps=2,
        )
        self.assertFalse(result.solved)

    def test_done_only_once_the_tests_actually_pass(self):
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertTrue(result.solved)

    def test_a_prose_reply_while_red_is_nudged_rather_than_accepted(self):
        _, llm = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_text("I think it is fine now."),
                assistant_text("still fine"),
            ],
            max_steps=3,
        )
        self.assertTrue(any(getattr(m, "content", None) == NUDGE for m in llm.calls[-1]))

    def test_a_write_invalidates_a_previously_green_result(self):
        """Otherwise: run tests, pass, then break them, and the verdict still says SOLVED."""
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": BUGGY}),
                assistant_text("done"),
            ]
        )
        self.assertFalse(result.solved, "a stale green result must not survive a write")

    def test_a_write_after_a_green_run_in_the_same_turn_ends_red(self):
        """One message, two calls, and the ORDER of the fold decides the verdict.

        An order-insensitive `tests_passed_after` passes every other test in this suite, so
        without this the fold could be "simplified" into reporting SOLVED for code that was
        never measured.
        """
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                # A different call in between, or the loop guard refuses the run_tests below as
                # a repeat and it never executes — which is what defeated the first version of
                # this test: no ExecResult was produced, so the fold order never mattered.
                assistant_tool_call("list_files", {}),
                # Green, then broken again, inside one turn.
                assistant_tool_calls(
                    [("run_tests", {}), ("write_file", {"path": "cart.py", "content": BUGGY})],
                    call_ids=("c1", "c2"),
                ),
                assistant_text("done"),
            ]
        )
        self.assertFalse(result.solved, "the write came last, so nothing has measured this code")
        self.assertEqual((self.tmp / "cart.py").read_text(), BUGGY, "the write did happen")

    def test_a_run_after_a_write_in_the_same_turn_ends_green(self):
        """The mirror image: measured last, so the measurement stands."""
        result, _ = self.run_with(
            [
                assistant_tool_calls(
                    [("write_file", {"path": "cart.py", "content": FIXED}), ("run_tests", {})],
                    call_ids=("c1", "c2"),
                ),
                assistant_text("done"),
            ]
        )
        self.assertTrue(result.solved)

    def test_a_turn_that_changes_nothing_leaves_a_green_verdict_alone(self):
        """`tests_passed_after` is seeded with the verdict so far, not with False.

        list_files neither measures nor modifies anything, so a green result stays green across
        it. Reseeding from False each turn would nudge an agent that had already succeeded.
        """
        result, _ = self.run_with(
            [
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("list_files", {}),
                assistant_text("done"),
            ]
        )
        self.assertTrue(result.solved)

    def test_the_agent_cannot_fake_success_by_rewriting_the_tests(self):
        result, _ = self.run_with(
            [
                assistant_tool_call(
                    "write_file",
                    {"path": "test_cart.py", "content": "import unittest\n"},
                ),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertFalse(result.solved)
        self.assertIn("assertEqual", (self.tmp / "test_cart.py").read_text())


class TestBudgetAndGuard(GraphTestCase):
    def test_the_step_budget_is_respected(self):
        result, llm = self.run_with(
            [assistant_tool_call("list_files", {}) for _ in range(10)], max_steps=3
        )
        self.assertEqual(result.steps_used, 3)
        self.assertEqual(llm.index, 3, "the graph must not exceed its budget")

    def test_an_identical_repeated_call_is_not_executed_again(self):
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("list_files", {}, call_id="c1"),
                assistant_tool_call("list_files", {}, call_id="c2"),
                assistant_text("done"),
            ],
            max_steps=3,
            tracer=tracer,
        )
        guarded = [e for e in tracer.events if "guarded" in e.detail]
        self.assertEqual(len(guarded), 1)

    def test_key_order_does_not_defeat_the_guard(self):
        """{"a":1,"b":2} and {"b":2,"a":1} are the same call."""
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call(
                    "write_file", {"path": "a.py", "content": "x = 1\n"}, call_id="c1"
                ),
                assistant_tool_call(
                    "write_file", {"content": "x = 1\n", "path": "a.py"}, call_id="c2"
                ),
                assistant_text("done"),
            ],
            max_steps=3,
            tracer=tracer,
        )
        self.assertTrue(any("guarded" in e.detail for e in tracer.events))

    def test_a_stuck_model_is_abandoned_rather_than_looping_to_the_budget(self):
        result, llm = self.run_with(
            [assistant_tool_call("list_files", {}, call_id=f"c{i}") for i in range(10)],
            max_steps=10,
        )
        self.assertFalse(result.solved)
        self.assertLessEqual(llm.index, 4)

    def test_the_second_repeat_warns_that_the_run_will_be_abandoned(self):
        """Only the first wording was pinned; the escalation could be deleted unnoticed."""
        tracer = Tracer()
        _, llm = self.run_with(
            [assistant_tool_call("list_files", {}, call_id=f"c{i}") for i in range(4)],
            max_steps=4,
            tracer=tracer,
        )
        answers = [str(m.content) for m in llm.calls[-1] if getattr(m, "tool_call_id", None)]
        self.assertTrue(
            any("abandoned" in a and "3" in a for a in answers),
            "the escalated observation names the consequence",
        )

    def test_making_progress_resets_the_guard(self):
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("list_files", {}, call_id="c1"),
                assistant_tool_call("run_tests", {}, call_id="c2"),
                assistant_tool_call("list_files", {}, call_id="c3"),
                assistant_text("done"),
            ],
            max_steps=4,
            tracer=tracer,
        )
        self.assertFalse(any("guarded" in e.detail for e in tracer.events))


class TestFanOutContract(GraphTestCase):
    """What the guard-node shape requires of its caller, and of a turn's tool_call_ids.

    The tool step is one `Send` task per pending call, which buys the guard a refusal that
    needs no synthetic message — and moves two guarantees out of any single node's reach.
    Both are enforced rather than documented, because both fail quietly.
    """

    def test_the_run_config_must_serialise_the_tool_step(self):
        """A caller that forgets `max_concurrency=1` is refused, not silently raced.

        `run_agent` always sets it. Anyone invoking the compiled graph directly can forget,
        and the symptom would otherwise be a rare false SOLVED on turns that write and test
        together — the single worst failure this project can have, and an intermittent one.
        """
        app = build_graph(
            FakeChatModel(replies=[assistant_tool_call("run_tests", {}), assistant_text("ok")]),
            self.tools,
            Tracer(),
            max_steps=2,
        )
        with self.assertRaises(RuntimeError) as ctx:
            app.invoke(initial_state(system_prompt(self.tools), "Fix it."))
        self.assertIn("max_concurrency=1", str(ctx.exception))

    def test_a_tool_call_id_reused_across_turns_still_dispatches(self):
        """The pending diff is scoped to the turn, so a reused id cannot swallow a call.

        The framework's own router collects answered ids from the WHOLE history, which assumes
        `tool_call_id`s are globally unique. They are the model's to choose. llm/fake.py reuses
        `call_1` by default, which is exactly the case that exposes it: with a history-wide
        diff, turn two dispatched nothing at all and the fold then reported an unanswered call.
        """
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("list_files", {}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=3,
            tracer=tracer,
        )
        executed = [e.detail for e in tracer.events if e.kind == "tool"]
        self.assertEqual(len(executed), 2, "both turns' calls ran despite the shared id")
        self.assertFalse(
            any("guarded" in detail for detail in executed),
            "neither call was refused — they are different calls",
        )


class TestToolNodeContract(GraphTestCase):
    def test_the_calls_in_one_turn_execute_one_at_a_time(self):
        """The oracle guarantee: a test run must never race a write in the same turn.

        ToolNode batches through a real ThreadPoolExecutor, so without `max_concurrency=1` the
        calls in one message run concurrently and `run_tests` can measure the file as it was
        BEFORE a write in the same turn — a green verdict for code that no longer exists.
        Message order is preserved either way, so nothing in the trace would look wrong.
        """
        timeline: list[str] = []
        slow_write = WriteFileTool(root=self.tmp)
        original = slow_write._run

        def traced(path: str, content: str):
            timeline.append("write-start")
            time.sleep(0.3)
            result = original(path=path, content=content)
            timeline.append("write-end")
            return result

        object.__setattr__(slow_write, "_run", traced)
        self.tools[2] = slow_write

        real_run = self.run_tests._run

        def traced_run():
            timeline.append("run-start")
            out = real_run()
            timeline.append("run-end")
            return out

        object.__setattr__(self.run_tests, "_run", traced_run)

        self.run_with(
            [
                assistant_tool_calls(
                    [("write_file", {"path": "cart.py", "content": FIXED}), ("run_tests", {})],
                    call_ids=("c1", "c2"),
                ),
                assistant_text("done"),
            ],
            max_steps=2,
        )
        self.assertEqual(
            timeline,
            ["write-start", "write-end", "run-start", "run-end"],
            "the tests must not begin until the write has finished",
        )

    def test_a_dropped_tool_answer_raises_rather_than_corrupting_the_next_request(self):
        """This invariant used to be an `assert`, which `python -O` removes.

        If ToolNode ever answers fewer calls than it was given, the API rejects the NEXT
        request — a turn away from the cause. Fail here instead, with the count.
        """
        with mock.patch.object(ToolNode, "invoke", return_value={"messages": []}):
            with self.assertRaises(RuntimeError) as ctx:
                self.run_with(
                    [assistant_tool_call("run_tests", {}), assistant_text("done")], max_steps=2
                )
        self.assertIn("must get exactly one reply", str(ctx.exception))
        self.assertIn("0 of 1", str(ctx.exception))


class TestCheckpointing(GraphTestCase):
    """State the framework can snapshot, which only works if the state holds everything."""

    def _run(self, app, llm, state=None):
        """Invoke `app` on thread "t". `state=None` starts a run; a partial dict resumes one.

        `max_concurrency=1` is not decoration. The tool step fans out one task per call, so
        serialising them is the run config's job and `guard_node` refuses to proceed without
        it — which is what these tests would otherwise discover as a confusing error rather
        than as the requirement it is. `run_agent` sets it for every ordinary run.
        """
        return app.invoke(
            initial_state(system_prompt(self.tools), "Fix it.") if state is None else state,
            config={
                "configurable": {"thread_id": "t"},
                "callbacks": [Tracer()],
                "max_concurrency": 1,
            },
        )

    def test_a_run_can_be_resumed_from_its_checkpoint(self):
        """The verdict has to live in the state, or a resumed solved task comes back unsolved.

        Two invocations against one saver: the fix is written in the first, the verifying test
        run happens in the second, and the verdict crosses the gap because it is checkpointed
        rather than held on a tool object that the second graph rebuilds empty.
        """
        saver = InMemorySaver()
        first = FakeChatModel(
            replies=[assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED})]
        )
        app = build_graph(first, self.tools, Tracer(), max_steps=1, checkpointer=saver)
        interrupted = self._run(app, first)
        self.assertFalse(interrupted["tests_passed"], "written but not yet verified")

        # A fresh model, a fresh graph, fresh tools — and `{"messages": []}` as the input, which
        # adds nothing and so leaves every checkpointed value in place. This is a resume, not a
        # second run: the history the model receives is the one from above.
        second = FakeChatModel(
            replies=[assistant_tool_call("run_tests", {}), assistant_text("done")]
        )
        resumed = build_graph(second, self.tools, Tracer(), max_steps=3, checkpointer=saver)
        final = self._run(resumed, second, state={"messages": []})

        self.assertTrue(final["tests_passed"], "the resumed run must see its own green suite")
        self.assertGreaterEqual(
            len(second.calls[0]),
            len(interrupted["messages"]),
            "the resumed model was sent the checkpointed history, not a bare prompt",
        )

        # Compared by content rather than by message equality: the checkpointer's serialiser
        # round-trips a frozen dataclass artifact back as a plain dict, so a replayed
        # ToolMessage is identical in everything the model sees but not `==` to the original.
        self.assertEqual(
            [m.content for m in final["messages"][: len(interrupted["messages"])]],
            [m.content for m in interrupted["messages"]],
        )
        # The step budget is per-invocation, so the resumed run counted its own turns on top.
        self.assertGreater(final["step"], interrupted["step"])

    def test_the_verdict_itself_survives_a_resume(self):
        """The test that would have caught the original bug, and the one I first got wrong.

        `test_a_run_can_be_resumed_from_its_checkpoint` re-runs the suite after resuming, so its
        green verdict is re-derived rather than restored — it would pass even if the verdict were
        not checkpointed at all. Here run 2 never calls a tool, so the ONLY possible source of
        `tests_passed` is the checkpoint.
        """
        saver = InMemorySaver()
        first = FakeChatModel(
            replies=[
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
            ]
        )
        app = build_graph(first, self.tools, Tracer(), max_steps=2, checkpointer=saver)
        self.assertTrue(self._run(app, first)["tests_passed"])

        second = FakeChatModel(replies=[assistant_text("already fixed")])
        resumed = build_graph(second, self.tools, Tracer(), max_steps=1, checkpointer=saver)
        final = self._run(resumed, second, state={"messages": []})

        self.assertTrue(final["tests_passed"], "the verdict was not carried across the resume")
        self.assertEqual(second.index, 1, "run 2 took exactly one turn")
        self.assertFalse(
            [m for m in second.calls[0] if getattr(m, "name", None) == "run_tests"][1:],
            "run 2 must not have re-measured anything of its own",
        )

    def test_every_step_is_recoverable_from_the_history(self):
        """`get_state_history` is what makes a run inspectable after the fact."""
        saver = InMemorySaver()
        llm = FakeChatModel(
            replies=[
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("write_file", {"path": "cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
            ]
        )
        app = build_graph(llm, self.tools, Tracer(), max_steps=3, checkpointer=saver)
        self._run(app, llm)
        config = {"configurable": {"thread_id": "t"}}
        # `.get`, not `[...]`: the history includes a snapshot taken before any node ran, whose
        # values are just the input.
        verdicts = [
            snapshot.values.get("tests_passed") for snapshot in app.get_state_history(config)
        ]
        self.assertIn(True, verdicts, "the green run is in the history")
        self.assertIn(False, verdicts, "so is the red one it started from")


class TestTracing(GraphTestCase):
    def test_every_model_turn_and_tool_call_is_recorded(self):
        tracer = Tracer()
        result, _ = self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=2,
            tracer=tracer,
        )
        self.assertEqual([e.kind for e in tracer.events], ["llm", "tool", "llm"])
        self.assertEqual(result.trace, tuple(tracer.events))

    def test_a_tool_line_reports_the_context_size_of_the_turn_that_asked_for_it(self):
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("run_tests", {}, prompt_tokens=1234),
                assistant_text("done"),
            ],
            max_steps=2,
            tracer=tracer,
        )
        tool_event = next(e for e in tracer.events if e.kind == "tool")
        self.assertEqual(tool_event.prompt_tokens, 1234)

    def test_the_absence_of_reasoning_is_visible(self):
        """Measured on Mellum2: a tool-calling turn carries no reasoning text at all."""
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ],
            max_steps=2,
            tracer=tracer,
        )
        self.assertIn("NO REASONING", tracer.events[0].detail)

    def test_reasoning_is_shown_when_the_model_does_emit_it(self):
        tracer = Tracer()
        self.run_with(
            [
                assistant_tool_calls([("run_tests", {})], text="Let me see what fails first."),
                assistant_text("done"),
            ],
            max_steps=2,
            tracer=tracer,
        )
        self.assertIn("Let me see what fails first.", tracer.events[0].detail)
