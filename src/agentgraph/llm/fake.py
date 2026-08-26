"""A scripted stand-in for the model, so the graph can be tested with nothing running.

This is why the test suite is fast, offline and deterministic: instead of mocking the graph's
internals, you hand it a *list of replies* and let the real graph execute against real tools in
a real temp directory. Nothing is patched — `FakeChatModel` is a real `BaseChatModel`, so the
graph cannot tell the difference.

A test reads like a screenplay of a model's turns:

    llm = FakeChatModel(replies=[
        assistant_tool_call("run_tests", {}, reasoning="Nothing has been measured yet."),
        assistant_tool_call("read_file", {"path": "shopcart/cart.py"}),
        assistant_tool_call("write_file", {"path": "shopcart/cart.py", "content": FIXED}),
        assistant_tool_call("run_tests", {}),
        assistant_text("Fixed the tax rounding."),
    ])

The tests really are red before that write and really are green after it, because the fake
replaces only the model — never the tools, the sandbox, or the graph.

Every builder takes an optional `reasoning=`, which is what makes the ReAct behaviour testable
offline. It has to go in the same place the real client puts it —
`additional_kwargs["reasoning_content"]`, see llm/client.py — because the graph and the tracer
read it from there. A fake that put reasoning anywhere else would let a broken agent pass:
`idle_turns`, the two nudges and the trace's `thinks` line would all be exercised against a
field the real model never populates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _extra(reasoning: str) -> dict[str, Any]:
    """The reasoning channel, or nothing at all.

    Empty rather than `{"reasoning_content": ""}` when there is no reasoning, so that a turn
    scripted without it is indistinguishable from a real non-thinking turn — `reasoning_of`
    treats a missing key and a blank one alike, but only one of them is what the wire produces.
    """
    return {"reasoning_content": reasoning} if reasoning else {}


def assistant_text(text: str, prompt_tokens: int = 10, reasoning: str = "") -> AIMessage:
    """A prose reply with no tool calls — the model talking rather than acting."""
    # The completion count is a stand-in. Nothing asserts on the value; it just has to be
    # plausible so the accounting in the graph has something to add up.
    return AIMessage(
        content=text,
        usage_metadata=_usage(prompt_tokens, len(text.split())),
        additional_kwargs=_extra(reasoning),
    )


def assistant_thinking(reasoning: str, prompt_tokens: int = 10) -> AIMessage:
    """A turn that reasoned and asked for NOTHING — the characteristic thinking-model failure.

    Empty `content` with a populated reasoning channel, which is exactly what the real client
    produces for a turn like this. The graph must not read it as an answer, must not end the
    run on it, and after MAX_IDLE_TURNS of them must abandon the run.
    """
    return assistant_text("", prompt_tokens=prompt_tokens, reasoning=reasoning)


def assistant_tool_call(
    name: str,
    arguments: dict[str, Any],
    call_id: str = "call_1",
    prompt_tokens: int = 10,
    reasoning: str = "",
) -> AIMessage:
    """A reply requesting one tool call, optionally having reasoned about it first."""
    return assistant_tool_calls(
        [(name, arguments)],
        call_ids=(call_id,),
        prompt_tokens=prompt_tokens,
        reasoning=reasoning,
    )


def assistant_tool_calls(
    calls: Sequence[tuple[str, dict[str, Any]]],
    call_ids: Sequence[str] | None = None,
    prompt_tokens: int = 10,
    text: str = "",
    reasoning: str = "",
) -> AIMessage:
    """A reply requesting SEVERAL tool calls in one turn, which a real model may do.

    The API permits any number of calls per assistant message, and requires an answer to every
    one of them: skip a `tool_call_id` and the next request is rejected. The graph therefore
    iterates over the calls — and without this builder there was no way to exercise that
    iteration with more than one element, so a graph answering only the first call would have
    passed every test in the suite.

        assistant_tool_calls([("run_tests", {}), ("list_files", {})], call_ids=("c1", "c2"))

    `call_ids` defaults to call_1..call_N. Ids must be distinct, since the whole point of an id
    is to tell two calls apart.
    """
    if call_ids is None:
        call_ids = [f"call_{index}" for index in range(1, len(calls) + 1)]
    ids = tuple(call_ids)
    assert len(ids) == len(calls), "one call id per call"
    assert len(set(ids)) == len(ids), "call ids must be distinct"

    return AIMessage(
        content=text,
        tool_calls=[
            {"name": name, "args": arguments, "id": call_id}
            for call_id, (name, arguments) in zip(ids, calls, strict=True)
        ],
        usage_metadata=_usage(prompt_tokens, 5 * len(calls)),
        additional_kwargs=_extra(reasoning),
    )


def unreadable_reply(detail: str = "Function arguments are not valid JSON.") -> Exception:
    """A scripted reply the client cannot parse at all — it RAISES instead of returning.

    This is what `ChatOllama` actually does with malformed tool-call arguments that did not
    arrive inside a dict: `_parse_json_string(..., skip=False)` raises rather than reporting an
    invalid call. Put one of these in a script and the graph's recovery path is exercised for
    real, exception and all — the alternative was trusting that a `try` block nobody had ever
    run would do the right thing.

    This is the only bad-JSON shape the agent can meet, which is why it is the only one the
    fake can script.
    """
    return OutputParserException(detail)


class FakeChatModel(BaseChatModel):
    """Scripted client so the agent graph is testable with no model running."""

    # `Any` rather than `AIMessage`, so a script can also contain an exception for the model to
    # raise instead of a message to return. See `unreadable_reply`.
    replies: list[Any]
    index: int = 0
    # Every history this model was called with, for tests that assert on what the graph
    # actually sent — that it is append-only, that the tool_call_id came back, and so on.
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "fake-agentgraph"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        """Ignore the tools on purpose: the replies are scripted, so nothing inspects schemas.

        Overridden because `BaseChatModel.bind_tools` raises NotImplementedError, and returning
        `self` (rather than a `RunnableBinding`) keeps `index` and `calls` observable from the
        test that built this object.
        """
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # An assert, not an IndexError, because the message is the diagnosis: if the graph asks
        # for more turns than the test scripted, the test's model of the graph is wrong. This
        # fires whenever a change makes the agent take an extra step.
        assert self.index < len(self.replies), (
            f"FakeChatModel script exhausted after {self.index} call(s); "
            "the agent asked for more turns than the test scripted"
        )
        # Snapshot the history at this turn, since the graph keeps appending.
        self.calls.append(list(messages))
        reply = self.replies[self.index]
        # Advanced BEFORE the raise, so a script with two unreadable replies in a row really
        # delivers two of them rather than the same one forever.
        self.index += 1
        if isinstance(reply, BaseException):
            raise reply
        assert isinstance(reply, AIMessage), f"scripted reply {self.index - 1} is not a message"
        return ChatResult(generations=[ChatGeneration(message=reply)])
