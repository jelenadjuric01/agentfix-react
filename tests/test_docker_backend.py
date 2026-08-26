"""The docker backend's command line. Asserted without a Docker daemon present."""

from __future__ import annotations

import unittest
from pathlib import Path

from agentfix.sandbox.docker_backend import DEFAULT_IMAGE, DockerBackend


class TestDockerArgv(unittest.TestCase):
    def setUp(self) -> None:
        self.argv = DockerBackend().build_argv(
            Path("/tmp/ws"), ("/usr/bin/python3", "-m", "unittest", "discover", "-q"), name="c1"
        )

    def _pair(self, flag: str) -> str:
        return self.argv[self.argv.index(flag) + 1]

    def test_the_container_is_removed_and_named(self):
        self.assertIn("--rm", self.argv)
        self.assertEqual(self._pair("--name"), "c1")

    def test_there_is_no_network(self):
        """The real difference from the subprocess backend."""
        self.assertEqual(self._pair("--network"), "none")

    def test_resources_are_capped(self):
        self.assertEqual(self._pair("--memory"), "512m")
        self.assertEqual(self._pair("--pids-limit"), "128")
        self.assertEqual(self._pair("--cpus"), "1")

    def test_privileges_are_dropped(self):
        self.assertEqual(self._pair("--user"), "runner")
        self.assertEqual(self._pair("--cap-drop"), "ALL")
        self.assertEqual(self._pair("--security-opt"), "no-new-privileges")

    def test_the_filesystem_is_immutable_apart_from_tmp(self):
        self.assertIn("--read-only", self.argv)
        self.assertEqual(self._pair("--tmpfs"), "/tmp")

    def test_the_workspace_is_mounted_read_only(self):
        """The file tools write on the host, so the container never needs write access."""
        self.assertEqual(self._pair("--volume"), "/tmp/ws:/work:ro")
        self.assertEqual(self._pair("--workdir"), "/work")

    def test_bytecode_writing_is_disabled(self):
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", self.argv)

    def test_the_host_interpreter_path_is_replaced_by_the_container_python(self):
        self.assertEqual(
            self.argv[self.argv.index(DEFAULT_IMAGE) + 1 :],
            ["python", "-m", "unittest", "discover", "-q"],
        )

    def test_container_names_are_unique_per_run(self):
        names = {DockerBackend()._container_name() for _ in range(50)}
        self.assertEqual(len(names), 50)
