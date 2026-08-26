"""The three file tools: list_files, read_file, write_file.

Each is a `BaseTool` subclass: LangChain derives the schema the model sees from `args_schema`,
and calls `_run` with the arguments already parsed and validated against it. That validation
replaces two of the hand-written checks in the no-framework edition — a missing argument and a
wrong type are now rejected before `_run` is entered.

What it does NOT replace is any of the four checks below. `resolve_in_root`, `is_test_path`,
`ast.parse` and `truncate` encode facts about *this* problem that no framework can infer:
these tools run in-process on the host, with no container, so validating before acting is the
only thing between a model-supplied path and your real filesystem.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from agentfix.tools.base import (
    MAX_FILE_READ_CHARS,
    NoArgs,
    WorkspaceChanged,
    truncate,
)

# Noise the model should never spend context on.
IGNORED_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache", ".mypy_cache"}

PROTECTED_HINT = (
    "Refused: {path} is part of the test suite. The tests are the specification — "
    "fix the source instead."
)


NEW_FILE_HINT = (
    "Refused: {path} is not one of this project's existing files. Fix a file that is "
    "already there rather than creating a new one."
)


class PathEscapeError(Exception):
    """Raised when a requested path resolves outside the task root."""


def resolve_in_root(root: Path, candidate: str) -> Path:
    """Resolve `candidate` under `root`, refusing anything that escapes it.

    This is the containment boundary for every file operation. Because the agent process
    itself is not sandboxed, this function is the only thing standing between a model-supplied
    path and your real filesystem — hence a raised exception rather than a warning.

    `.resolve()` is what makes it work: it collapses "..", follows symlinks and produces an
    absolute path, so "../../../../etc/passwd" and a symlink pointing outside the workspace
    are both caught. Comparing the raw strings would not catch either.
    """
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    # Allowed if it IS the root, or if the root is one of its parents.
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise PathEscapeError(f"{candidate} resolves outside the task root")
    return resolved


def is_test_path(root: Path, target: Path) -> bool:
    """`run_tests` is the only oracle, so letting the agent rewrite the tests is a bypass.

    Without this, an agent that cannot fix the bug can still turn the suite green by deleting
    the failing assertion — and the graph's `is_done` check would report SOLVED. That was
    reproducible before this check existed.
    """
    try:
        # Both sides resolved. `WriteFileTool` always passes resolved output, so this is
        # belt-and-braces — but the `except` below fails OPEN, treating an unrecognised path as
        # "not a test file", and that is the wrong direction to be sloppy in for the only check
        # protecting the agent's oracle from the agent.
        relative = target.resolve().relative_to(root.resolve())
    except ValueError:
        # Not under the root at all. resolve_in_root already refused it; say "not a test file"
        # rather than crashing.
        return False
    # Both comparisons are case-INSENSITIVE, and that is not pedantry. macOS ships a
    # case-insensitive filesystem, so "Tests/TEST_CART.PY" addresses the very same inode as
    # "tests/test_cart.py" — while a case-sensitive check calls it an ordinary source file and
    # waves it through. Reproduced on macOS: the agent overwrote the protected suite with a
    # trivially passing test and the run reported SOLVED with the bug untouched.
    if any(part.lower() == "tests" for part in relative.parts):
        return True
    return relative.name.lower().startswith("test_")


def relative_files(root: Path) -> list[str]:
    """Every .py file under `root`, sorted, as paths relative to it.

    Relative and sorted on purpose: absolute paths would leak the temp directory name into the
    model's context and change on every run, which wastes tokens and makes traces impossible
    to compare.
    """
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        found.append(str(path.relative_to(root)))
    return found


class ListFilesTool(BaseTool):
    """Orientation: what files exist. Takes no arguments."""

    name: str = "list_files"
    description: str = "List the Python files in the project, relative to the project root."
    args_schema: type[BaseModel] = NoArgs

    # A pydantic field, not a plain attribute: `BaseTool` is a pydantic model, so per-instance
    # state has to be declared to be settable. Each run has a different workspace.
    root: Path

    def _run(self) -> str:
        files = relative_files(self.root)
        if not files:
            # An empty project is not an error, so report it as a normal observation.
            return "The project contains no Python files."
        return truncate("\n".join(files))


class ReadArgs(BaseModel):
    path: str = Field(description="Path relative to the project root")


class ReadFileTool(BaseTool):
    """Read one file. The model is told to read only files the failure implicates."""

    name: str = "read_file"
    description: str = "Read the contents of one file in the project."
    args_schema: type[BaseModel] = ReadArgs

    root: Path

    def _run(self, path: str) -> str:
        try:
            target = resolve_in_root(self.root, path)
        except PathEscapeError:
            # Refusals are observations, not exceptions — the model can try another path.
            return f"Refused: {path} is outside the project root."

        if not target.is_file():
            # Listing what does exist turns a typo into a one-turn recovery.
            available = ", ".join(relative_files(self.root))
            return f"No such file: {path}. Files in this project: {available}"

        return truncate(target.read_text(encoding="utf-8"), MAX_FILE_READ_CHARS)


class WriteArgs(BaseModel):
    path: str = Field(description="Path relative to the project root")
    content: str = Field(description="The complete new file contents")


class WriteFileTool(BaseTool):
    """Replace one file wholesale. Deliberately not a diff tool.

    Production agents usually edit via diffs. At 12B, models reliably emit invalid unified
    diffs — drifting line numbers, mismatched context — and burn their step budget failing to
    apply a patch instead of fixing anything. A full-file rewrite is far more reliable at this
    model size. The general lesson: a tool contract the model cannot satisfy looks exactly
    like a broken agent.
    """

    name: str = "write_file"
    description: str = (
        "Replace the entire contents of one file. Provide the complete new file, not a diff."
    )
    args_schema: type[BaseModel] = WriteArgs

    # A successful write returns a `WorkspaceChanged` artifact alongside the text the model
    # reads. That artifact is what invalidates the last test result: it described the code as
    # it was. Every other branch below returns `None`, so a refused write invalidates nothing.
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    root: Path

    # The relative paths this tool may write, taken from the task's pristine template.
    #
    # An allow-list rather than more refusal rules, because "does this path look dangerous" is
    # an unwinnable game. Two reproduced escapes made the case: writing `unittest.py` at the
    # workspace root shadows the stdlib module that `run_tests` executes (a red suite then
    # exits 0, so `is_done` reports SOLVED with the bug untouched), and a `.pth` file under a
    # workspace-relative site-packages path runs arbitrary code at interpreter startup. Neither
    # goes anywhere near a name `is_test_path` inspects.
    #
    # The agent's job is to repair a file that already exists, so it never needs to create one.
    # Saying exactly that closes every "write a NEW file that changes what the tests do" route
    # at once, including the ones nobody has thought of yet. `None` disables the check, which
    # only tests use.
    allowed: frozenset[str] | None = None

    def _run(self, path: str, content: str) -> tuple[str, WorkspaceChanged | None]:
        try:
            target = resolve_in_root(self.root, path)
        except PathEscapeError:
            return f"Refused: {path} is outside the project root.", None

        # Guard the oracle: the tests are the specification, so they are not writable.
        if is_test_path(self.root, target):
            return PROTECTED_HINT.format(path=path), None

        # Then the allow-list. Compared on the resolved path's own relative form, so a
        # case-variant alias of a real file ("Tests/TEST_CART.PY") does not match the template
        # entry it aliases, and is refused here even on a case-insensitive filesystem.
        if self.allowed is not None:
            relative = str(target.relative_to(self.root.resolve()))
            if relative not in self.allowed:
                return NEW_FILE_HINT.format(path=path), None

        # Parse before writing. `ast.parse` compiles the text without executing any of it, so
        # it is a syntax check with no side effects. Rejecting a broken file up front gives a
        # precise error ("line 12: unexpected indent") instead of leaving the project
        # unimportable and making the next `run_tests` fail for a confusing reason.
        try:
            ast.parse(content)
        except SyntaxError as error:
            return (
                f"Not written — the content has a syntax error on line "
                f"{error.lineno}: {error.msg}",
                None,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        # Report the size rather than echoing the content: the model just sent it, and
        # repeating it back would double its cost in context for no new information.
        return f"Wrote {len(content)} characters to {path}.", WorkspaceChanged(path=path)
