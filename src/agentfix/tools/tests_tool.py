"""The run_tests tool — the agent's only oracle, and the only sandboxed tool.

This is the most consequential tool in the project. The graph's stop condition consults the
result of this tool and nothing else, so a run ends successfully only because the tests
actually passed — never because the model announced it was finished.

The tool itself is stateless. The verdict travels back to the graph as the ToolMessage's
`artifact`: `response_format="content_and_artifact"` lets one `_run` return two things at
once — the text the model reads, and a structured value only the graph reads. See
agent/graph.py's `tests_passed_after` for the other end of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from langchain_core.tools import BaseTool

from agentfix.sandbox.base import ExecResult, ExecutionBackend
from agentfix.tools.base import NoArgs


class RunTestsTool(BaseTool):
    """Runs the task's test command through an execution backend and reports the result."""

    name: str = "run_tests"
    # The model is told outright that this is the source of truth. It is also *asked*, never
    # told: nothing hands the agent the failing test output up front, so discovering the
    # failure is part of the task.
    description: str = (
        "Run the project's test suite and return the result. This is the source of truth."
    )
    args_schema: type[BaseModel] = NoArgs

    # Two return values from one call: `content` for the model, `artifact` for the graph. The
    # ExecResult goes back untouched rather than being re-parsed out of the text — the model's
    # copy is prose, and prose is not something a stop condition should be reading.
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    root: Path
    command: tuple[str, ...]
    # Injected rather than constructed here, so tests can pass a fake backend and the
    # subprocess/docker choice stays the caller's (runner.py calls get_backend()).
    backend: Any
    timeout_s: int = 10

    # `ExecutionBackend` is a Protocol and `ExecResult` a dataclass; neither is something
    # pydantic can validate, so they have to be allowed explicitly.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def execution_backend(self) -> ExecutionBackend:
        """`backend` typed as Any for pydantic's sake; read it back with the real type."""
        backend: ExecutionBackend = self.backend
        return backend

    def _run(self) -> tuple[str, ExecResult]:
        result = self.execution_backend.run(self.root, self.command, timeout_s=self.timeout_s)

        # Note this returns a normal observation even when the tests fail: the *tool* worked.
        # Failing tests are the information the agent needs, not a tool error. The headline is
        # prepended because unittest buries the verdict at the end of its report, and a 12B
        # model reading 2,000 characters of traceback does better when the answer is first.
        headline = "All tests passed." if result.passed else "Tests failed."
        return f"{headline}\n\n{result.output}".strip(), result
