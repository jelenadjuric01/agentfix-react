"""What a task is, and the disposable workspace every run gets."""

from __future__ import annotations

import json
import stat
import sys

from agentfix.tasks.loader import DEFAULT_PROMPT, load_task, workspace
from tests.support import TempDirTestCase


class TestLoadTask(TempDirTestCase):
    def _task_dir(self, meta: dict) -> None:
        (self.tmp / "repo").mkdir(exist_ok=True)
        (self.tmp / "task.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_reads_every_field(self):
        self._task_dir(
            {
                "task_id": "demo",
                "test_command": ["-m", "unittest", "discover", "-q"],
                "expected_failures": ["test_x"],
                "prompt": "Fix the bug.",
            }
        )
        task = load_task(self.tmp)
        self.assertEqual(task.task_id, "demo")
        self.assertEqual(task.expected_failures, ("test_x",))
        self.assertEqual(task.prompt, "Fix the bug.")
        self.assertEqual(task.template_dir, self.tmp / "repo")

    def test_a_minimal_task_json_still_loads(self):
        self._task_dir({})
        task = load_task(self.tmp)
        self.assertEqual(task.task_id, self.tmp.name)
        self.assertEqual(task.prompt, DEFAULT_PROMPT)
        self.assertEqual(task.test_command[1:], ("-m", "unittest", "discover", "-q"))

    def test_a_flag_first_command_is_pinned_to_this_interpreter(self):
        """ "python" on the sandbox PATH is not necessarily the project's virtualenv."""
        self._task_dir({"test_command": ["-m", "unittest"]})
        self.assertEqual(load_task(self.tmp).test_command[0], sys.executable)

    def test_a_command_naming_its_own_program_is_left_alone(self):
        self._task_dir({"test_command": ["/usr/bin/make", "test"]})
        self.assertEqual(load_task(self.tmp).test_command, ("/usr/bin/make", "test"))


class TestWorkspace(TempDirTestCase):
    def _task(self):
        from agentfix.tasks.loader import Task

        template = self.tmp / "repo"
        template.mkdir()
        (template / "a.py").write_text("original\n", encoding="utf-8")
        return Task("t", self.tmp, template, ("true",), (), "p")

    def test_the_copy_is_writable_and_the_template_is_untouched(self):
        task = self._task()
        with workspace(task) as work_dir:
            (work_dir / "a.py").write_text("rewritten\n", encoding="utf-8")
        self.assertEqual((task.template_dir / "a.py").read_text(), "original\n")

    def test_a_read_only_template_still_produces_a_writable_copy(self):
        """Workshop fixtures are often handed out read-only; the copy must not inherit that."""
        task = self._task()
        (task.template_dir / "pkg").mkdir()
        (task.template_dir / "pkg" / "b.py").write_text("original\n", encoding="utf-8")
        for path in (
            task.template_dir / "pkg" / "b.py",
            task.template_dir / "a.py",
            task.template_dir / "pkg",
            task.template_dir,
        ):
            path.chmod(0o555 if path.is_dir() else 0o444)
        try:
            with workspace(task) as work_dir:
                (work_dir / "a.py").write_text("rewritten\n", encoding="utf-8")
                # A rename into the directory, which is what write_file actually does.
                scratch = work_dir / "pkg" / "b.py.tmp"
                scratch.write_text("rewritten\n", encoding="utf-8")
                scratch.replace(work_dir / "pkg" / "b.py")
                self.assertEqual((work_dir / "pkg" / "b.py").read_text(), "rewritten\n")
                captured = work_dir
        finally:
            for path in (task.template_dir, task.template_dir / "pkg"):
                path.chmod(0o755)

        # The read-only template is unchanged, and the copy was still cleaned up.
        self.assertEqual((task.template_dir / "a.py").read_text(), "original\n")
        self.assertEqual(stat.S_IMODE((task.template_dir / "a.py").stat().st_mode), 0o444)
        self.assertFalse(captured.exists())

    def test_the_copy_is_deleted_on_the_way_out(self):
        with workspace(self._task()) as work_dir:
            captured = work_dir
        self.assertFalse(captured.exists())

    def test_the_copy_is_deleted_even_when_the_body_raises(self):
        """Without the try/finally, every failed run would leak a copy of the project."""
        captured = None
        with self.assertRaises(RuntimeError):
            with workspace(self._task()) as work_dir:
                captured = work_dir
                raise RuntimeError("tool crashed")
        self.assertIsNotNone(captured)
        self.assertFalse(captured.exists())
