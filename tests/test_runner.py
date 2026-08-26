"""The wiring. Two safety properties are established here, not in the graph."""

from __future__ import annotations

from pathlib import Path

from agentfix.llm.fake import FakeChatModel, assistant_text, assistant_tool_call
from agentfix.runner import solve_task
from tests.support import TempDirTestCase

FIXED = (
    "from shopcart.pricing import with_tax\n\n\n"
    "def subtotal(prices: list[float]) -> float:\n    return sum(prices)\n\n\n"
    "def total_with_tax(prices: list[float]) -> float:\n"
    "    return with_tax(subtotal(prices))\n"
)
SHOPCART = Path("tasks/workshop/01-shopcart")


class TestSolveTask(TempDirTestCase):
    def test_the_real_wiring_solves_a_real_fixture_with_a_scripted_model(self):
        llm = FakeChatModel(
            replies=[
                assistant_tool_call("run_tests", {}),
                assistant_tool_call("read_file", {"path": "shopcart/cart.py"}),
                assistant_tool_call("write_file", {"path": "shopcart/cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("Fixed the tax rounding."),
            ]
        )
        result = solve_task(SHOPCART, llm=llm, max_steps=5)
        self.assertTrue(result.solved)
        self.assertEqual(result.task_id, "01-shopcart")

    def test_the_pristine_fixture_is_byte_identical_afterwards(self):
        """The agent rewrites whole files; the next run must start from the same bug."""
        source = SHOPCART / "repo" / "shopcart" / "cart.py"
        before = source.read_bytes()
        llm = FakeChatModel(
            replies=[
                assistant_tool_call("write_file", {"path": "shopcart/cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        solve_task(SHOPCART, llm=llm, max_steps=3)
        self.assertEqual(source.read_bytes(), before)

    def test_the_tools_are_bound_to_the_workspace_not_the_repo(self):
        """A read of a repo-relative path must resolve inside the disposable copy."""
        llm = FakeChatModel(
            replies=[
                assistant_tool_call("list_files", {}),
                assistant_text("done"),
            ]
        )
        result = solve_task(SHOPCART, llm=llm, max_steps=2)
        listing = next(e.detail for e in result.trace if e.name == "list_files")
        self.assertIn("shopcart/cart.py", listing)
        self.assertNotIn(str(SHOPCART.resolve()), listing)

    def test_a_write_invalidates_the_last_test_result(self):
        """write_file reports WorkspaceChanged; without it a stale green result survives."""
        llm = FakeChatModel(
            replies=[
                assistant_tool_call("write_file", {"path": "shopcart/cart.py", "content": FIXED}),
                assistant_tool_call("run_tests", {}),
                assistant_tool_call(
                    "write_file",
                    {
                        "path": "shopcart/cart.py",
                        "content": "def total_with_tax(p):\n    return 0\n",
                    },
                ),
                assistant_text("done"),
            ]
        )
        self.assertFalse(solve_task(SHOPCART, llm=llm, max_steps=4).solved)
