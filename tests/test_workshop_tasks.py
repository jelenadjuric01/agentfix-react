"""Every shipped fixture must start red, with exactly the failures it claims.

The cheapest way for this workshop to waste twenty minutes is a task that is already green, or
red for the wrong reason. These run the real suites, so they catch it.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

from agentfix.config import REPO_ROOT
from agentfix.tasks.loader import load_task, workspace

TASK_DIRS = sorted(p.parent for p in (REPO_ROOT / "tasks" / "workshop").glob("*/task.json"))


class TestWorkshopFixtures(unittest.TestCase):
    def test_there_are_fixtures_to_run(self):
        self.assertGreaterEqual(len(TASK_DIRS), 3)

    def test_each_fixture_starts_red_with_the_failures_it_declares(self):
        for task_dir in TASK_DIRS:
            with self.subTest(task=task_dir.name):
                task = load_task(task_dir)
                with workspace(task) as work_dir:
                    completed = subprocess.run(
                        list(task.test_command),
                        cwd=work_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                combined = completed.stdout + completed.stderr
                self.assertNotEqual(completed.returncode, 0, "fixture is already green")
                self.assertTrue(task.expected_failures, "a fixture must declare its failures")
                for name in task.expected_failures:
                    self.assertIn(name, combined)

    def test_each_fixture_has_a_discoverable_test_package(self):
        """Without tests/__init__.py, `unittest discover` finds nothing and exits 5."""
        for task_dir in TASK_DIRS:
            with self.subTest(task=task_dir.name):
                tests_dir = task_dir / "repo" / "tests"
                if tests_dir.is_dir():
                    self.assertTrue((tests_dir / "__init__.py").exists())

    def test_discovery_actually_finds_more_than_zero_tests(self):
        for task_dir in TASK_DIRS:
            with self.subTest(task=task_dir.name):
                task = load_task(task_dir)
                with workspace(task) as work_dir:
                    completed = subprocess.run(
                        [sys.executable, "-m", "unittest", "discover"],
                        cwd=work_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                self.assertNotIn("NO TESTS RAN", completed.stdout + completed.stderr)
