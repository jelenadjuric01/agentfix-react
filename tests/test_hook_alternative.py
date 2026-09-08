"""The two LangGraph behaviours `agent/graph.py`'s shape depends on.

The agent borrows the framework's own shape for the guard: `create_react_agent` wires
`post_model_hook` by computing

    pending = [c for c in last_ai_message.tool_calls if c["id"] not in answered_tool_call_ids]

and dispatching one task per pending call — so answering a call is what refuses it.
`guard_node` and `route_after_guard` are that router, reproduced, because the hook itself is a
`create_react_agent` argument while `Send` is public to anyone.

These tests are why that shape has the two extra pieces it has. Neither asserts anything about
our code; both pin down third-party behaviour the design rests on, which is the kind of claim
that belongs in a test rather than a comment:

  - `max_concurrency=1` throttles Send fan-out, so the oracle guarantee survives one task per
    call — but only from the RUN config, which is why `guard_node` refuses to proceed without
    it instead of trusting the caller.
  - One task per call means several writers per superstep, and a reducer-free key refuses that
    — which is why the verdict fold lives in `fold_node` and not with the guard.

Run:  uv run python -m unittest tests.test_hook_alternative -v
"""

from __future__ import annotations

import operator
import threading
import time
import unittest
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

SLEEP = 0.05


class FanState(TypedDict):
    calls: list[str]
    log: Annotated[list[str], operator.add]


class VerdictState(TypedDict):
    calls: list[str]
    log: Annotated[list[str], operator.add]
    # Deliberately WITHOUT a reducer, exactly as AgentState.tests_passed is. See state.py for
    # why: a reducer is handed (current, incoming) and cannot tell "the suite went green" from
    # "this call changed the workspace, so the verdict is void".
    tests_passed: bool


def _fan(state: Any) -> Any:
    return [Send("work", {"name": name}) for name in state["calls"]]


class TestSendFanOutRespectsMaxConcurrency(unittest.TestCase):
    """The oracle guarantee survives the hook's shape. This one is good news.

`run_agent` sets `max_concurrency=1` on the run config, and `guard_node` refuses to
    dispatch without it, because `ToolNode` otherwise runs a turn's calls in a real thread pool
    — which can let `run_tests` measure the workspace as it was before a `write_file` in the
    same turn, and that is a false SOLVED.

    The worry worth settling was whether one task per call escapes that config entirely. It
    does not: `max_concurrency` throttles the fan-out too, which is what makes the whole shape
    usable here.
    """

    def _run(self, config: dict[str, Any]) -> tuple[int, list[str]]:
        live = 0
        peak = 0
        lock = threading.Lock()

        def work(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(SLEEP)
            with lock:
                live -= 1
            return {"log": [payload["name"]]}

        graph = StateGraph(FanState)
        graph.add_node("start", lambda state: {})
        graph.add_node("work", work)
        graph.add_edge(START, "start")
        graph.add_conditional_edges("start", _fan, ["work"])
        graph.add_edge("work", END)
        app = graph.compile()

        result = app.invoke({"calls": ["a", "b", "c"], "log": []}, config=config)
        return peak, result["log"]

    def test_fan_out_runs_in_parallel_by_default(self):
        peak, log = self._run({})
        self.assertEqual(peak, 3, "three Sends ran concurrently — the default, and the hazard")
        self.assertEqual(log, ["a", "b", "c"], "message order is preserved either way")

    def test_max_concurrency_serialises_the_fan_out(self):
        peak, log = self._run({"max_concurrency": 1})
        self.assertEqual(peak, 1, "one at a time, so the oracle guarantee holds under Send too")
        self.assertEqual(log, ["a", "b", "c"])


class TestFanOutBreaksAReducerFreeKey(unittest.TestCase):
    """And this is the bill. One task per call makes the tool step a CONCURRENT writer.

Fan the calls out and each task wants to write `tests_passed` in one superstep, which
    LangGraph refuses for any key without a reducer. This is the whole reason `fold_node`
    exists: the fold happens once, after the tool step, where it is a single writer again.

    The alternative was a reducer on `tests_passed`, which state.py argues against on its own
    merits — a reducer is handed (current, incoming) and cannot tell "the suite went green"
    from "the workspace changed, so the verdict is void".
    """

    def _run(self, calls: list[str]) -> dict[str, Any]:
        def work(payload: dict[str, Any]) -> dict[str, Any]:
            # The shape `fold_node` exists to avoid: reply AND fold in one task.
            return {"log": [payload["name"]], "tests_passed": True}

        graph = StateGraph(VerdictState)
        graph.add_node("start", lambda state: {})
        graph.add_node("work", work)
        graph.add_edge(START, "start")
        graph.add_conditional_edges("start", _fan, ["work"])
        graph.add_edge("work", END)
        app = graph.compile()

        return app.invoke(
            {"calls": calls, "log": [], "tests_passed": False},
            config={"max_concurrency": 1},
        )

    def test_one_call_per_turn_is_fine(self):
        result = self._run(["a"])
        self.assertTrue(result["tests_passed"], "a single writer, so nothing to reconcile")

    def test_two_calls_in_one_turn_cannot_both_write_the_verdict(self):
        from langgraph.errors import InvalidUpdateError

        with self.assertRaises(InvalidUpdateError) as ctx:
            self._run(["a", "b"])
        self.assertIn("tests_passed", str(ctx.exception))
        self.assertIn("only one value per step", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
