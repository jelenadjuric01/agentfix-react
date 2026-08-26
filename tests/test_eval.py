"""The eval harness: pass@1 arithmetic, and HumanEvalFix -> task directory."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentfix.agent.graph import AgentResult
from agentfix.llm.fake import FakeChatModel, assistant_text
from agentfix.eval.humanevalfix import (
    HumanEvalFixRow,
    as_unittest_module,
    load_vendored_rows,
    write_task_dir,
)
from agentfix.eval.runner import EvalReport, crashed, evaluate
from agentfix.tasks.loader import load_task
from tests.support import TempDirTestCase


def result(task_id: str, solved: bool, steps: int = 4, peak: int = 100) -> AgentResult:
    return AgentResult(task_id, solved, steps, 1000, 200, 1.0, (), peak)


class TestOneCrashDoesNotEndTheSuite(TempDirTestCase):
    """Ten minutes of a real model must not be thrown away by one exception."""

    def _task(self, name: str, prompt: str = "Fix it.") -> None:
        (self.tmp / name / "repo").mkdir(parents=True)
        (self.tmp / name / "task.json").write_text(
            json.dumps({"task_id": name, "prompt": prompt}), encoding="utf-8"
        )

    def test_a_failing_task_is_recorded_and_the_rest_still_run(self):
        self._task("good")
        missing = self.tmp / "not-a-task"  # no task.json — load_task will raise

        llm = FakeChatModel(replies=[assistant_text("nothing to do"), assistant_text("nor here")])
        with redirect_stdout(io.StringIO()):
            results = evaluate([missing, self.tmp / "good"], llm=llm, max_steps=1)

        self.assertEqual([r.task_id for r in results], ["not-a-task", "good"])
        self.assertFalse(results[0].solved)
        self.assertEqual(results[0].trace[0].kind, "error", "the reason is kept, not swallowed")

    def test_the_crash_row_names_the_exception(self):
        row = crashed(Path("/tasks/17-broken"), RuntimeError("model went away"))
        self.assertEqual(row.task_id, "17-broken")
        self.assertFalse(row.solved)
        self.assertIn("model went away", row.trace[0].detail)
        self.assertIn("RuntimeError", row.trace[0].name)


class TestEvalReport(unittest.TestCase):
    def test_pass_at_1_is_the_solved_fraction(self):
        report = EvalReport("s", (result("a", True), result("b", False)))
        self.assertEqual(report.pass_at_1, 0.5)

    def test_an_empty_suite_scores_zero_rather_than_dividing_by_zero(self):
        self.assertEqual(EvalReport("s", ()).pass_at_1, 0.0)

    def test_peak_prompt_tokens_is_the_max_across_the_suite(self):
        report = EvalReport("s", (result("a", True, peak=900), result("b", True, peak=1500)))
        self.assertEqual(report.peak_prompt_tokens, 1500)

    def test_peak_of_an_empty_suite_does_not_raise(self):
        self.assertEqual(EvalReport("s", ()).peak_prompt_tokens, 0)

    def test_json_drops_the_trace_but_keeps_the_numbers(self):
        payload = EvalReport("s", (result("a", True),)).to_json()
        self.assertEqual(payload["pass_at_1"], 1.0)
        self.assertNotIn("trace", payload["results"][0])
        self.assertEqual(payload["results"][0]["task_id"], "a")

    def test_the_table_reports_every_task_and_the_summary(self):
        table = EvalReport("s", (result("a", True), result("b", False))).format_table()
        self.assertIn("a", table)
        self.assertIn("b", table)
        self.assertIn("pass@1 = 0.50", table)


class TestVendoredSubset(unittest.TestCase):
    def test_the_committed_subset_loads_without_any_optional_dependency(self):
        rows = load_vendored_rows()
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(r.entry_point for r in rows))

    def test_a_missing_key_fails_at_load_rather_than_much_later(self):
        with self.assertRaises(TypeError):
            HumanEvalFixRow(**{"task_id": "x"})  # type: ignore[arg-type]


class TestAsUnittestModule(unittest.TestCase):
    ROW = HumanEvalFixRow(
        task_id="HumanEval/1",
        buggy_code="def add(a, b):\n    return a - b\n",
        tests=(
            "from candidate import add\n\n"
            "def check(add):\n    assert add(1, 2) == 3, 'Test 1'\n\n"
            "check(add)\n"
        ),
        entry_point="add",
    )

    def test_the_bare_invocation_is_replaced_by_a_test_case(self):
        """A bare assert at import time is a collection error, not a test failure."""
        module = as_unittest_module(self.ROW)
        self.assertIn("import unittest", module)
        self.assertIn("class TestCandidate(unittest.TestCase):", module)
        self.assertNotIn("\ncheck(add)\n", module)
        self.assertIn("        check(add)", module)

    def test_the_benchmarks_own_assertions_are_untouched(self):
        """Rewriting them would change what is being measured."""
        self.assertIn("assert add(1, 2) == 3, 'Test 1'", as_unittest_module(self.ROW))

    def test_the_generated_module_is_valid_python(self):
        import ast

        ast.parse(as_unittest_module(self.ROW))


class TestWriteTaskDir(TempDirTestCase):
    ROW = TestAsUnittestModule.ROW

    def test_produces_a_directory_the_loader_can_read(self):
        task_dir = write_task_dir(self.ROW, self.tmp)
        task = load_task(task_dir)
        self.assertEqual(task.task_id, "humaneval-1")
        self.assertIn("add", task.prompt)
        self.assertEqual(task.test_command[1:], ("-m", "unittest", "discover", "-q"))

    def test_the_test_file_sits_at_the_root_so_discover_finds_it(self):
        """Only subdirectories need to be importable packages."""
        repo = write_task_dir(self.ROW, self.tmp) / "repo"
        self.assertTrue((repo / "test_candidate.py").is_file())
        self.assertTrue((repo / "candidate.py").is_file())

    def test_the_generated_task_json_is_valid(self):
        payload = json.loads((write_task_dir(self.ROW, self.tmp) / "task.json").read_text())
        self.assertEqual(payload["task_id"], "humaneval-1")

    def test_the_generated_suite_really_is_red_for_the_buggy_code(self):
        """The starts-red property comes from the benchmark, so verify it end to end."""
        import subprocess
        import sys

        repo = write_task_dir(self.ROW, self.tmp) / "repo"
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-q"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
