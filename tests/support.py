"""Shared helpers for the test suite.

Not a test module — the name deliberately does not start with `test_`, so `unittest discover`
does not try to collect it.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agentfix.sandbox.base import ExecResult
from agentfix.tasks.loader import Task

# The `llm`-marked tests of the no-framework edition were selected by a pytest marker plus a
# `--all` flag wired up in conftest.py. unittest has no markers, so the same opt-in is an
# environment variable and a decorator. Same property: the whole suite passes offline, and the
# tests that need a live Ollama are opt-in rather than opt-out.
LLM_TESTS_ENABLED = os.environ.get("AGENTFIX_LLM_TESTS") == "1"

requires_ollama = unittest.skipUnless(
    LLM_TESTS_ENABLED,
    "needs a running Ollama with the model from the README; set AGENTFIX_LLM_TESTS=1",
)

PYTHON_UNITTEST = ("python", "-m", "unittest", "discover", "-q")


class TempDirTestCase(unittest.TestCase):
    """A TestCase with a throwaway directory, the unittest equivalent of pytest's tmp_path.

    `enterContext` (3.11+) registers the context manager's cleanup with the test, so the
    directory is removed however the test ends — the same guarantee `workspace` gives the agent.
    """

    def setUp(self) -> None:
        super().setUp()
        # `.resolve()` is not decoration. On macOS this hands back /var/folders/..., which is a
        # symlink to /private/var/folders/..., while every check in tools/fs.py compares against
        # `.resolve()` output. Without it those comparisons silently disagree.
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()


class FakeBackend:
    """An execution backend that returns a canned result and records what it was asked to run.

    Satisfies the `ExecutionBackend` protocol structurally, without inheriting from it — which
    is the whole point of Protocol, and why `run_tests` can be tested with no subprocess.
    """

    def __init__(self, result: ExecResult | None = None) -> None:
        self.result = result or ExecResult(passed=False, output="1 failed", duration_s=0.01)
        self.calls: list[tuple[Path, tuple[str, ...], int]] = []

    def run(self, workspace: Path, command: tuple[str, ...], timeout_s: int = 10) -> ExecResult:
        self.calls.append((workspace, command, timeout_s))
        return self.result


def make_task(root: Path, task_id: str = "t", prompt: str = "Fix it.") -> Task:
    """A Task pointing at `root`, for tests that need one without a task.json on disk."""
    return Task(
        task_id=task_id,
        root=root,
        template_dir=root,
        test_command=PYTHON_UNITTEST,
        expected_failures=(),
        prompt=prompt,
    )
