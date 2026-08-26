"""The four validations in tools/fs.py, which are the agent's only containment."""

from __future__ import annotations

from pathlib import Path

from agentfix.tools.base import TRUNCATION_MARKER, WorkspaceChanged
from agentfix.tools.fs import (
    ListFilesTool,
    PathEscapeError,
    ReadFileTool,
    WriteFileTool,
    is_test_path,
    resolve_in_root,
)
from tests.support import TempDirTestCase


class TestResolveInRoot(TempDirTestCase):
    def test_a_plain_relative_path_resolves_inside(self):
        self.assertEqual(resolve_in_root(self.tmp, "a.py"), (self.tmp / "a.py").resolve())

    def test_the_root_itself_is_allowed(self):
        self.assertEqual(resolve_in_root(self.tmp, "."), self.tmp.resolve())

    def test_dot_dot_escape_is_refused(self):
        with self.assertRaises(PathEscapeError):
            resolve_in_root(self.tmp, "../../../../etc/passwd")

    def test_an_absolute_path_outside_is_refused(self):
        with self.assertRaises(PathEscapeError):
            resolve_in_root(self.tmp, "/etc/passwd")

    def test_a_symlink_pointing_out_is_refused(self):
        """String comparison would miss this; `.resolve()` is what catches it."""
        outside = self.tmp.parent / "outside_target"
        outside.mkdir(exist_ok=True)
        (self.tmp / "link").symlink_to(outside)
        with self.assertRaises(PathEscapeError):
            resolve_in_root(self.tmp, "link/secret.py")


class TestIsTestPath(TempDirTestCase):
    def test_a_file_in_a_tests_directory_is_protected(self):
        self.assertTrue(is_test_path(self.tmp, self.tmp / "tests" / "test_cart.py"))

    def test_a_test_prefixed_file_at_the_root_is_protected(self):
        self.assertTrue(is_test_path(self.tmp, self.tmp / "test_candidate.py"))

    def test_ordinary_source_is_not_protected(self):
        self.assertFalse(is_test_path(self.tmp, self.tmp / "shopcart" / "cart.py"))

    def test_a_path_outside_the_root_is_not_called_a_test(self):
        self.assertFalse(is_test_path(self.tmp, Path("/etc/passwd")))


class TestListFiles(TempDirTestCase):
    def test_lists_python_files_relative_and_sorted(self):
        (self.tmp / "pkg").mkdir()
        (self.tmp / "pkg" / "b.py").write_text("")
        (self.tmp / "a.py").write_text("")
        (self.tmp / "notes.txt").write_text("ignored")
        out = ListFilesTool(root=self.tmp).invoke({})
        self.assertEqual(out.splitlines(), ["a.py", "pkg/b.py"])

    def test_ignores_noise_directories(self):
        (self.tmp / "__pycache__").mkdir()
        (self.tmp / "__pycache__" / "x.py").write_text("")
        self.assertIn("no Python files", ListFilesTool(root=self.tmp).invoke({}))

    def test_an_empty_project_is_an_observation_not_an_error(self):
        self.assertIn("no Python files", ListFilesTool(root=self.tmp).invoke({}))


class TestReadFile(TempDirTestCase):
    def test_reads_a_file(self):
        (self.tmp / "a.py").write_text("x = 1\n")
        self.assertEqual(ReadFileTool(root=self.tmp).invoke({"path": "a.py"}), "x = 1\n")

    def test_a_missing_file_lists_what_does_exist(self):
        (self.tmp / "real.py").write_text("")
        out = ReadFileTool(root=self.tmp).invoke({"path": "typo.py"})
        self.assertIn("No such file", out)
        self.assertIn("real.py", out)

    def test_an_escape_is_refused_as_an_observation_not_an_exception(self):
        out = ReadFileTool(root=self.tmp).invoke({"path": "../../etc/passwd"})
        self.assertIn("Refused", out)

    def test_a_large_file_is_truncated_with_a_visible_marker(self):
        (self.tmp / "big.py").write_text("# " + "x" * 9000)
        out = ReadFileTool(root=self.tmp).invoke({"path": "big.py"})
        self.assertTrue(out.endswith(TRUNCATION_MARKER))


class TestWriteFile(TempDirTestCase):
    def test_writes_and_reports_the_size(self):
        out = WriteFileTool(root=self.tmp).invoke({"path": "a.py", "content": "x = 1\n"})
        self.assertEqual((self.tmp / "a.py").read_text(), "x = 1\n")
        self.assertIn("6 characters", out)

    def test_refuses_to_rewrite_the_test_suite(self):
        """The tests are the specification. Without this an agent can fake a green run."""
        (self.tmp / "tests").mkdir()
        (self.tmp / "tests" / "test_cart.py").write_text("original")
        out = WriteFileTool(root=self.tmp).invoke(
            {"path": "tests/test_cart.py", "content": "# deleted the assertion"}
        )
        self.assertIn("Refused", out)
        self.assertEqual((self.tmp / "tests" / "test_cart.py").read_text(), "original")

    def test_a_syntax_error_is_reported_and_nothing_is_written(self):
        out = WriteFileTool(root=self.tmp).invoke({"path": "a.py", "content": "def f(:\n"})
        self.assertIn("syntax error", out)
        self.assertFalse((self.tmp / "a.py").exists())

    def test_an_escape_is_refused(self):
        out = WriteFileTool(root=self.tmp).invoke({"path": "../evil.py", "content": "x = 1"})
        self.assertIn("Refused", out)
        self.assertFalse((self.tmp.parent / "evil.py").exists())

    def _message(self, tool: WriteFileTool, path: str, content: str):
        call = {"name": "write_file", "args": {"path": path, "content": content}}
        return tool.invoke({**call, "id": "c1", "type": "tool_call"})

    def test_a_successful_write_reports_the_workspace_changed(self):
        """That artifact is what invalidates the last test result, in the graph's state."""
        message = self._message(WriteFileTool(root=self.tmp), "a.py", "x = 1\n")
        self.assertIsInstance(message.artifact, WorkspaceChanged)
        self.assertEqual(message.artifact.path, "a.py")

    def test_a_rejected_write_reports_no_change(self):
        """A refusal changed nothing, so it must not invalidate a green test result."""
        tool = WriteFileTool(root=self.tmp)
        self.assertIsNone(self._message(tool, "b.py", "def f(:\n").artifact)
        self.assertIsNone(self._message(tool, "../evil.py", "x = 1\n").artifact)
        (self.tmp / "tests").mkdir()
        (self.tmp / "tests" / "test_x.py").write_text("original")
        self.assertIsNone(self._message(tool, "tests/test_x.py", "x = 1\n").artifact)

    def test_creates_missing_parent_directories(self):
        WriteFileTool(root=self.tmp).invoke({"path": "deep/nested/a.py", "content": "x = 1\n"})
        self.assertTrue((self.tmp / "deep" / "nested" / "a.py").is_file())
