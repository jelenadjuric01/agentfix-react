"""Observability: one record per model turn and per tool call.

Small, but the thing that makes the agent debuggable. An agent that prints only SOLVED or
NOT SOLVED gives you nothing to act on; a trace shows which turn went wrong, how much
context each turn cost, and where the time went. `agentgraph solve --verbose` prints it live.

This is a `BaseCallbackHandler`, which is the framework's own observability seam: it is handed
to the graph once, as `config={"callbacks": [tracer]}`, and LangChain then calls the hooks
below as the model and the tools run. The graph's nodes contain no tracing code at all — they
are state transitions and nothing else, which is the whole reason to prefer this over an
observer the nodes have to remember to call.

## What changed in the ReAct edition, and why it had to

The previous edition read the model's reasoning off `AIMessage.content`, because that is where
it would have been: a model with no thinking mode puts everything it says in one place. Against
the Thinking checkpoint that is simply the wrong field. Reasoning arrives on its own channel
(`additional_kwargs["reasoning_content"]`, see llm/client.py) and `content` is routinely EMPTY
on a turn that reasoned for three hundred tokens and then called a tool.

So reasoning is now a field of its own on `TraceEvent`, printed on its own line, and
`(NO REASONING)` is computed from the channel it actually lives on — where it now means
something real: this model skipped thinking on this turn.

Two things the hooks cannot see, because they never happened: a call the loop guard refused to
execute, and a tool call whose JSON never parsed. No tool ran, so no tool callback fires. The
graph reports those itself with `note`, and that split is the honest one — the framework
observes what the framework did.

"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult

# Long enough to read a sentence of model output, short enough that one event stays on
# one line. A readable 20-line trace beats a faithful 500-line dump of file contents.
DETAIL_CLIP = 250

# Reasoning gets a longer clip than everything else, because it is the one field where the
# interesting part is often not in the first sentence — a model talks itself INTO the wrong
# file over several sentences, and clipping at 250 hides the turn.
REASONING_CLIP = 400


@dataclass(frozen=True)
class TraceEvent:
    """One thing that happened, at a known step. Frozen: history is recorded, not rewritten."""

    step: int  # which graph iteration, 1-based
    kind: str  # "llm" or "tool"
    name: str  # "assistant", or the tool's name
    detail: str  # the model's answer and actions, or the tool's output
    prompt_tokens: int  # context size at this point — watch it grow across a run
    latency_s: float

    # The model's reasoning for this turn, on its own field rather than mixed into `detail`.
    # Defaulted so every existing construction site — including the tool events, which have no
    # reasoning by definition — stays valid without being told about it.
    reasoning: str = ""


def reasoning_of(message: AIMessage) -> str:
    """The model's thinking for this turn, or "" if it did not think.

    One `.get`, and the single most important line in this file, because it is the only place
    that knows WHERE reasoning lives. `reasoning=True` on the client puts it here (see
    llm/client.py); with `reasoning` unset the same text would be inline in `content` wrapped
    in `<think>` tags, and everything downstream — the trace, `write_file`'s "complete file
    contents", the next prompt — would be quietly polluted with it.
    """
    return str(message.additional_kwargs.get("reasoning_content") or "").strip()


def describe(message: AIMessage) -> str:
    """One line summarising what a model turn DID — its actions and its answer.

    Deliberately not its reasoning: that has its own field now. What is left here is the
    decision the turn produced, which is the thing you scan a trace for.

    `(NO REASONING)` survives from the previous edition but now means what it says. There it
    was printed whenever `content` was empty, which against a thinking model is most turns.
    Here it is printed only when the model genuinely did not think before acting.
    """
    text = (message.text or "").strip()
    if message.tool_calls:
        names = ", ".join(call["name"] or "unknown" for call in message.tool_calls)
        summary = f"calls {names}"
        if text:
            return f"{summary} -- {text}"
        # No answer text is normal and unremarkable when the model reasoned instead; it is
        # only worth flagging when there was no reasoning either.
        return summary if reasoning_of(message) else f"{summary} -- (NO REASONING)"
    return text


def prompt_tokens_of(message: AIMessage) -> int:
    """The context size the server reported for this turn, or 0 if it reported none."""
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    return int(usage.get("input_tokens", 0))


class Tracer(BaseCallbackHandler):
    """Collects events from LangChain's callbacks, and optionally prints them as they happen."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.events: list[TraceEvent] = []

        # Which model turn we are in. Counted here rather than read from the graph's state
        # because a callback is handed no state — and the number is the same one, since a turn
        # begins precisely when the model is called.
        self.step = 0

        # The context size of the current turn, so a tool line reports the same `ctx` as the
        # model line that asked for it. Remembered rather than passed around: tool callbacks
        # know nothing about the message that requested the call.
        self.turn_prompt_tokens = 0

        # run_id -> (start time, name). Keyed by run because callbacks are not guaranteed to
        # nest, and a tool that fails never reaches on_tool_end.
        #
        # The name is stored because `on_tool_error` is NOT told which tool raised: LangChain
        # passes it only run_id, parent_run_id, tags and tool_call_id. `on_tool_start` is the
        # one hook that gets the name, in `serialized`. Without stashing it here, the single
        # trace line you most need to identify reads "tool:tool".
        self._started: dict[UUID, tuple[float, str]] = {}

    # --- what the framework observes --------------------------------------------------------

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        """A model turn is starting. This is the only place `step` advances."""
        self.step += 1
        self._mark(kwargs.get("run_id"), "assistant")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # Indexing defensively, and recording when there is nothing to record. A raise in here
        # would be swallowed: `BaseCallbackHandler.raise_error` is False, so LangChain logs a
        # WARNING on a logger nothing in this project configures and carries on — the run
        # survives and the event silently disappears from the trace.
        generations = response.generations
        reply = generations[0][0] if generations and generations[0] else None
        message = getattr(reply, "message", None)
        if not isinstance(message, AIMessage):
            # A turn that produced no assistant message still happened, and `step` has already
            # advanced. Say so, and clear the token count rather than letting this turn's tool
            # lines inherit the PREVIOUS turn's context size — a trace that is quietly wrong
            # about context growth is worse than one with a gap in it.
            self.turn_prompt_tokens = 0
            self.note("llm", "assistant", "no assistant message in the model's reply")
            return
        reply_message = message
        self.turn_prompt_tokens = prompt_tokens_of(reply_message)
        self.record(
            TraceEvent(
                step=self.step,
                kind="llm",
                name="assistant",
                detail=describe(reply_message),
                prompt_tokens=self.turn_prompt_tokens,
                latency_s=self._elapsed(kwargs.get("run_id")),
                reasoning=reasoning_of(reply_message),
            )
        )

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """The only hook told which tool this is — `serialized["name"]`. Remember it."""
        self._mark(kwargs.get("run_id"), str(serialized.get("name") or "tool"))

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        remembered = self._name(kwargs.get("run_id"))
        name = output.name if isinstance(output, ToolMessage) else remembered
        detail = str(output.content) if isinstance(output, ToolMessage) else str(output)
        self.record(
            TraceEvent(
                step=self.step,
                kind="tool",
                name=str(name or "tool"),
                detail=detail,
                prompt_tokens=self.turn_prompt_tokens,
                latency_s=self._elapsed(kwargs.get("run_id")),
            )
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """A tool raised. ToolNode turns this into an observation; the trace should say so."""
        run_id = kwargs.get("run_id")
        # Read the name BEFORE `_elapsed`, which pops the entry. Sequenced explicitly rather
        # than relying on argument evaluation order.
        name = self._name(run_id)
        self.record(
            TraceEvent(
                step=self.step,
                kind="tool",
                name=name,
                detail=f"raised {type(error).__name__}: {error}",
                prompt_tokens=self.turn_prompt_tokens,
                latency_s=self._elapsed(run_id),
            )
        )

    # --- what only the graph knows ----------------------------------------------------------

    def note(self, kind: str, name: str, detail: str) -> None:
        """Record something the graph decided rather than something the framework ran.

        Callers are all cases where the framework saw nothing: a call the loop guard refused,
        a reply the client could not parse (the model call raised, so `on_llm_end` never
        fired), and a turn abandoned for thinking without ever acting. Latency is 0.0 because
        nothing happened — that is the point.
        """
        self.record(TraceEvent(self.step, kind, name, detail, self.turn_prompt_tokens, 0.0))

    # --- recording ---------------------------------------------------------------------------

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self.verbose:
            # → "we asked the model", ← "something came back".
            marker = "→" if event.kind == "llm" else "←"
            detail = event.detail.replace("\n", " ")[:DETAIL_CLIP]
            print(
                f"  step {event.step} {marker} {event.kind}:{event.name}  "
                f"[ctx {event.prompt_tokens} tok, {event.latency_s:.1f}s]  {detail}"
            )
            if event.reasoning:
                # Indented under the turn it belongs to and labelled, so a run reads as
                # thought-then-action rather than as two unrelated lines.
                thought = event.reasoning.replace("\n", " ")[:REASONING_CLIP]
                print(f"          thinks  {thought}")

    def as_json(self) -> list[dict[str, Any]]:
        """Serialise for the eval report. `asdict` turns each dataclass into a plain dict."""
        return [asdict(event) for event in self.events]

    # --- timing ------------------------------------------------------------------------------

    def _mark(self, run_id: UUID | None, name: str) -> None:
        if run_id is not None:
            self._started[run_id] = (time.time(), name)

    def _name(self, run_id: UUID | None) -> str:
        """What the start callback said this run was, without consuming the entry."""
        entry = self._started.get(run_id) if run_id is not None else None
        return "tool" if entry is None else entry[1]

    def _elapsed(self, run_id: UUID | None) -> float:
        """Seconds since the matching start callback. `pop` so the map cannot grow forever."""
        entry = self._started.pop(run_id, None) if run_id is not None else None
        return 0.0 if entry is None else round(time.time() - entry[0], 2)
