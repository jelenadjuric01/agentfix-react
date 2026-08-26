"""Wiring: turns a task directory into a finished run.

Short, and worth reading closely — this is where the separate pieces are assembled, and where
two of the project's safety properties are actually established. Read it after agent/graph.py.

    task dir -> load_task -> workspace copy -> tools bound to that copy -> run_agent
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from agentfix.agent.graph import MAX_STEPS, AgentResult, run_agent
from agentfix.agent.trace import Tracer
from agentfix.sandbox.base import get_backend
from agentfix.tasks.loader import load_task, workspace
from agentfix.tools.fs import ListFilesTool, ReadFileTool, WriteFileTool, relative_files
from agentfix.tools.tests_tool import RunTestsTool


def solve_task(
    task_dir: Path,
    llm: BaseChatModel | None = None,
    verbose: bool = False,
    max_steps: int = MAX_STEPS,
) -> AgentResult:
    """Run the agent on one task, from a fresh copy, and return what happened.

    `llm=None` means "make a real one". Tests pass a `FakeChatModel` here instead, which is how
    the whole suite runs the real wiring with no model process anywhere.
    """
    if llm is None:
        # Imported inside the function, not at module scope, so that importing this module —
        # and therefore the test suite — never constructs a network client or reads the
        # environment. Nothing here requires a running Ollama until you actually ask for one.
        from agentfix.llm.client import make_chat_model

        llm = make_chat_model()

    task = load_task(Path(task_dir))

    # Read from the PRISTINE template, not the workspace copy, so the set cannot grow during a
    # run: an agent that managed to create a file could otherwise then write to it.
    writable = frozenset(relative_files(task.template_dir))

    # Every run gets a disposable copy, and it is deleted when this block exits by any route.
    with workspace(task) as work_dir:
        # Each tool is constructed with `work_dir` bound to it. That is why the graph's nodes
        # need no workspace argument: by this point the workspace is baked into the tools.
        #
        # A flat list with no wiring between the entries, which is newer than it looks. The
        # tools used to be ordered — run_tests built first so that WriteFileTool could be
        # handed its `invalidate` method — and run_tests then passed to the graph a second
        # time so the stop condition could read it. Both went away when the tools started
        # reporting what they did as ToolMessage artifacts: run_tests returns its ExecResult,
        # write_file returns a WorkspaceChanged, and the graph folds the two into its state.
        tools = [
            ListFilesTool(root=work_dir),
            ReadFileTool(root=work_dir),
            WriteFileTool(root=work_dir, allowed=writable),
            RunTestsTool(
                root=work_dir, command=task.test_command, backend=get_backend(), timeout_s=30
            ),
        ]

        return run_agent(task, llm, tools, max_steps=max_steps, tracer=Tracer(verbose))
