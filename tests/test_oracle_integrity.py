"""The agent must not be able to make the tests pass without fixing the bug.

`run_tests` is the only oracle, so any write that changes what the suite *does* — rather than
what the code does — is a privilege escalation. Every case here was a reproduced escape before
the checks that stop it existed, so these are regression tests, not hypotheticals.
"""

from __future__ import annotations

import subprocess
import sys

from agentfix.llm.fake import FakeChatModel, assistant_text, assistant_tool_call
from agentfix.runner import solve_task
from agentfix.tools.fs import WriteFileTool, is_test_path, relative_files
from tests.support import TempDirTestCase
from tests.test_runner import SHOPCART

PASSING_SUITE = (
    "import unittest\n\n\nclass TestCart(unittest.TestCase):\n"
    "    def test_ok(self):\n        self.assertTrue(True)\n"
)


class TestCaseInsensitiveBypass(TempDirTestCase):
    """macOS is case-insensitive: "Tests/TEST_CART.PY" is the same inode as the real suite."""

    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "tests").mkdir()
        (self.tmp / "tests" / "test_cart.py").write_text("original\n", encoding="utf-8")
        self.allowed = frozenset(relative_files(self.tmp))

    def test_is_test_path_is_case_insensitive(self):
        for candidate in ["tests/test_cart.py", "Tests/TEST_CART.PY", "TESTS/Test_Cart.py"]:
            with self.subTest(path=candidate):
                self.assertTrue(is_test_path(self.tmp, self.tmp / candidate))

    def test_a_case_variant_write_cannot_reach_the_test_file(self):
        out = WriteFileTool(root=self.tmp, allowed=self.allowed).invoke(
            {"path": "Tests/TEST_CART.PY", "content": PASSING_SUITE}
        )
        self.assertIn("Refused", out)
        self.assertEqual((self.tmp / "tests" / "test_cart.py").read_text(), "original\n")


class TestRunnerShadowing(TempDirTestCase):
    """`python -m unittest` puts the workspace first on sys.path."""

    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "cart.py").write_text("def total():\n    return 0\n", encoding="utf-8")
        self.allowed = frozenset(relative_files(self.tmp))

    def test_the_stdlib_test_runner_cannot_be_shadowed(self):
        out = WriteFileTool(root=self.tmp, allowed=self.allowed).invoke(
            {"path": "unittest.py", "content": "import sys\n\nsys.exit(0)\n"}
        )
        self.assertIn("Refused", out)
        self.assertFalse((self.tmp / "unittest.py").exists())

    def test_shadowing_really_would_have_forged_the_oracle(self):
        """Proves the check above is load-bearing rather than guarding nothing."""
        (self.tmp / "tests").mkdir()
        (self.tmp / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (self.tmp / "tests" / "test_x.py").write_text(
            "import unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_red(self):\n        self.assertEqual(1, 2)\n",
            encoding="utf-8",
        )
        argv = [sys.executable, "-m", "unittest", "discover", "-q"]
        red = subprocess.run(argv, cwd=self.tmp, capture_output=True, check=False)
        self.assertNotEqual(red.returncode, 0, "the suite must start red")

        (self.tmp / "unittest.py").write_text("import sys\n\nsys.exit(0)\n", encoding="utf-8")
        forged = subprocess.run(argv, cwd=self.tmp, capture_output=True, check=False)
        self.assertEqual(forged.returncode, 0, "shadowing forges a pass — hence the allow-list")

    def test_a_startup_hook_cannot_be_planted(self):
        """A .pth file under a workspace-relative site-packages runs code before any test."""
        out = WriteFileTool(root=self.tmp, allowed=self.allowed).invoke(
            {"path": ".local/lib/python3.12/site-packages/evil.pth", "content": "import os\n"}
        )
        self.assertIn("Refused", out)


class TestAllowList(TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "cart.py").write_text("x = 1\n", encoding="utf-8")
        self.allowed = frozenset(relative_files(self.tmp))

    def test_an_existing_file_is_still_writable(self):
        """The fix must not stop the agent doing its actual job."""
        out = WriteFileTool(root=self.tmp, allowed=self.allowed).invoke(
            {"path": "cart.py", "content": "x = 2\n"}
        )
        self.assertIn("Wrote", out)
        self.assertEqual((self.tmp / "cart.py").read_text(), "x = 2\n")

    def test_no_allow_list_means_no_check(self):
        out = WriteFileTool(root=self.tmp).invoke({"path": "new.py", "content": "x = 1\n"})
        self.assertIn("Wrote", out)


class TestEndToEnd(TempDirTestCase):
    def test_the_agent_still_solves_a_real_fixture_with_the_allow_list_active(self):
        """Parity check: the hardening must not change the agent's legitimate behaviour."""
        fixed = (
            "from shopcart.pricing import with_tax\n\n\n"
            "def subtotal(prices: list[float]) -> float:\n    return sum(prices)\n\n\n"
            "def total_with_tax(prices: list[float]) -> float:\n"
            "    return with_tax(subtotal(prices))\n"
        )
        llm = FakeChatModel(
            replies=[
                assistant_tool_call("write_file", {"path": "shopcart/cart.py", "content": fixed}),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertTrue(solve_task(SHOPCART, llm=llm, max_steps=3).solved)

    def test_the_agent_cannot_forge_a_pass_through_the_real_wiring(self):
        llm = FakeChatModel(
            replies=[
                assistant_tool_call(
                    "write_file", {"path": "unittest.py", "content": "import sys\nsys.exit(0)\n"}
                ),
                assistant_tool_call("run_tests", {}),
                assistant_text("done"),
            ]
        )
        self.assertFalse(solve_task(SHOPCART, llm=llm, max_steps=3).solved)
