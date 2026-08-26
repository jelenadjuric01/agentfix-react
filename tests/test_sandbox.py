"""The subprocess backend, and the backend switch."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

from agentfix.sandbox.base import get_backend
from agentfix.sandbox.docker_backend import DockerBackend
from agentfix.sandbox.subprocess_backend import SubprocessBackend
from tests.support import TempDirTestCase

UNITTEST_CMD = (sys.executable, "-m", "unittest", "discover", "-q")


class TestSubprocessBackend(TempDirTestCase):
    def _write_test(self, body: str, name: str = "test_x.py") -> None:
        (self.tmp / name).write_text(body, encoding="utf-8")

    def test_a_passing_suite_reports_passed(self):
        self._write_test(
            "import unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_ok(self):\n        self.assertTrue(True)\n"
        )
        self.assertTrue(SubprocessBackend().run(self.tmp, UNITTEST_CMD, timeout_s=30).passed)

    def test_a_failing_suite_reports_not_passed_and_keeps_the_output(self):
        self._write_test(
            "import unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_no(self):\n        self.assertEqual(1, 2)\n"
        )
        result = SubprocessBackend().run(self.tmp, UNITTEST_CMD, timeout_s=30)
        self.assertFalse(result.passed)
        self.assertIn("test_no", result.output)

    def test_discovering_no_tests_is_not_a_pass(self):
        """unittest exits 5 here. If that read as success, every task would report SOLVED."""
        result = SubprocessBackend().run(self.tmp, UNITTEST_CMD, timeout_s=30)
        self.assertFalse(result.passed)

    def test_a_hanging_test_times_out_as_a_result_rather_than_an_exception(self):
        self._write_test(
            "import unittest, time\n\n\nclass T(unittest.TestCase):\n"
            "    def test_hang(self):\n        time.sleep(30)\n"
        )
        result = SubprocessBackend().run(self.tmp, UNITTEST_CMD, timeout_s=2)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.passed)
        self.assertIn("TIMEOUT", result.output)

    def test_output_is_truncated_with_a_marker(self):
        self._write_test(
            "import unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_loud(self):\n        print('x' * 50000)\n        self.assertEqual(1, 2)\n"
        )
        result = SubprocessBackend(max_output_chars=500).run(self.tmp, UNITTEST_CMD, timeout_s=30)
        self.assertLess(len(result.output), 800)
        self.assertIn("truncated", result.output)

    def test_the_child_environment_is_stripped(self):
        """No API keys, no PYTHONPATH that could shadow the task's own modules."""
        self._write_test(
            "import os, unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_env(self):\n"
            "        self.assertIsNone(os.environ.get('AGENTFIX_SECRET'))\n"
        )
        with mock.patch.dict(os.environ, {"AGENTFIX_SECRET": "leaked"}):
            self.assertTrue(SubprocessBackend().run(self.tmp, UNITTEST_CMD, timeout_s=30).passed)


class TestBackendSelection(unittest.TestCase):
    def test_defaults_to_subprocess(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(get_backend(), SubprocessBackend)

    def test_the_environment_variable_selects_docker(self):
        with mock.patch.dict(os.environ, {"AGENTFIX_SANDBOX": "docker"}):
            self.assertIsInstance(get_backend(), DockerBackend)

    def test_an_explicit_argument_wins_over_the_environment(self):
        with mock.patch.dict(os.environ, {"AGENTFIX_SANDBOX": "docker"}):
            self.assertIsInstance(get_backend("subprocess"), SubprocessBackend)

    def test_a_typo_fails_loudly_rather_than_silently_weakening_isolation(self):
        with self.assertRaises(ValueError):
            get_backend("dokcer")
