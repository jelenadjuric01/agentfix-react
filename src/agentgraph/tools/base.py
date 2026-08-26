"""What a tool is here, and the limits on what one may return.

This file is much shorter than its no-framework counterpart, and the missing part is the
interesting part. There, `ToolRegistry` had two jobs — describe the tools to the model
(`schemas`) and run the model's requested call (`dispatch`) — and `dispatch` was the trust
boundary, converting five kinds of model mistake into observations.

Both jobs are now the framework's:

    schemas()  -> `llm.bind_tools([...])` derives them from each tool's `args_schema`
    dispatch() -> `langgraph.prebuilt.ToolNode`

What survives here is the part no framework can know for you: how many bytes of a tool's
output the model can afford to read. See agent/graph.py for the two failure paths ToolNode
does *not* cover, and what this project does about them.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

# Every byte a tool returns is a byte of context the model must re-read on every later turn,
# so tool output is capped. These limits are the difference between a run that fits in a 16k
# context and one that silently truncates mid-history.
MAX_TOOL_OUTPUT_CHARS = 2000
MAX_FILE_READ_CHARS = 4000  # a file read gets a larger budget than a directory listing
TRUNCATION_MARKER = "\n[...truncated]"


class NoArgs(BaseModel):
    """The argument schema for a tool that takes no arguments.

    An empty model rather than None: `bind_tools` needs *something* to convert into the
    `{"type": "object", "properties": {}}` the model is shown.
    """


@dataclass(frozen=True)
class WorkspaceChanged:
    """Artifact returned by any tool that modified the workspace.

    A tool returns two things: `content`, which the model reads, and `artifact`, which only
    the graph reads (`response_format="content_and_artifact"`). This is the artifact that says
    "whatever the tests last reported is now describing code that no longer exists."

    Typed rather than a bare bool, and returned by the *tool* rather than wired up by the
    caller, because the previous arrangement — runner.py passing `run_tests.invalidate` as a
    write callback — only worked for the one pair of tools someone remembered to connect. Any
    future tool that edits the workspace gets the invalidation by returning this.
    """

    path: str


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Cap `text`, leaving a visible marker so the model knows it did not see everything."""
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARKER
