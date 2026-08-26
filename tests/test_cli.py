"""The command line. Parsing and exit codes only — no model, no network."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agentfix import __version__
from agentfix.agent.graph import MAX_STEPS, AgentResult
from agentfix.cli import build_parser, main


class TestParser(unittest.TestCase):
    def test_solve_takes_a_path_and_defaults_to_the_graphs_own_budget(self):
        args = build_parser().parse_args(["solve", "tasks/workshop/01-shopcart"])
        self.assertEqual(args.task_dir, Path("tasks/workshop/01-shopcart"))
        self.assertEqual(args.max_steps, MAX_STEPS)
        self.assertFalse(args.verbose)

    def test_eval_defaults_to_the_workshop_suite_and_a_small_limit(self):
        args = build_parser().parse_args(["eval"])
        self.assertEqual(args.suite, "workshop")
        self.assertEqual(args.limit, 3)

    def test_an_unknown_suite_is_rejected(self):
        # argparse prints its usage to stderr before exiting; swallowed so the suite's own
        # output stays clean.
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["eval", "--suite", "nonsense"])


class TestMain(unittest.TestCase):
    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_version(self):
        code, out, _ = self._run(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(__version__, out)

    def test_no_subcommand_prints_help_rather_than_failing_silently(self):
        code, out, _ = self._run([])
        self.assertEqual(code, 0)
        self.assertIn("usage", out.lower())

    def test_a_bad_argument_returns_a_code_instead_of_killing_the_process(self):
        code, _, _ = self._run(["eval", "--limit", "not-a-number"])
        self.assertEqual(code, 2)

    def test_the_exit_code_follows_the_verdict(self):
        """So `agentfix solve ... && echo ok` behaves sensibly and CI can gate on it."""
        solved = AgentResult("t", True, 4, 100, 20, 1.0, (), 100)
        with mock.patch("agentfix.runner.solve_task", return_value=solved):
            code, out, _ = self._run(["solve", "tasks/workshop/01-shopcart"])
        self.assertEqual(code, 0)
        self.assertIn("SOLVED", out)

    def test_an_unsolved_task_exits_non_zero(self):
        unsolved = AgentResult("t", False, 10, 100, 20, 1.0, (), 100)
        with mock.patch("agentfix.runner.solve_task", return_value=unsolved):
            code, out, _ = self._run(["solve", "tasks/workshop/01-shopcart"])
        self.assertEqual(code, 1)
        self.assertIn("NOT SOLVED", out)

    def test_a_failed_run_reports_a_legible_error_not_a_traceback(self):
        """A broken setup is the common case, so it gets a diagnosis and not a stack trace."""
        boom = ConnectionError("cannot reach http://localhost:11434")
        with mock.patch("agentfix.runner.solve_task", side_effect=boom):
            code, _, err = self._run(["solve", "tasks/workshop/01-shopcart"])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)
        self.assertIn("ConnectionError", err)
        self.assertIn("doctor", err)

    def test_the_summary_reports_how_many_turns_reasoned(self):
        """The headline number of this edition, so it belongs on the one line always printed."""
        solved = AgentResult("t", True, 4, 100, 20, 1.0, (), 100, reasoning_turns=3)
        with mock.patch("agentfix.runner.solve_task", return_value=solved):
            _, out, _ = self._run(["solve", "tasks/workshop/01-shopcart"])
        self.assertIn("thinks=3/4", out)
