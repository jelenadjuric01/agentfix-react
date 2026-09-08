"""The agent itself. If you read one file in this project, read this one.

Three editions of the same agent now exist. The first was a `for` loop. The second was that
loop as a LangGraph graph, driven by an Instruct model that only ever acted. This is the third:
the same graph, driven by a model that *reasons before it acts*.

The three things that decide whether it works at all are unchanged:

1. a bounded number of steps         — an agent with no cap is an unbounded wait and bill
2. a stop condition based on reality — `is_done`, which believes the test suite rather than
                                       the model's opinion of its own work
3. a loop guard                      — because a stuck model repeats itself forever

## What reasoning changed

Not the shape of the graph. Reasoning is not a node, and there is no "think" step — the model
thinks and acts in the SAME turn, which is what ReAct actually means. `reasoning=True` on the
client hands the thinking back on its own channel and the graph carries on as before.

What it changed is the two decisions above that touch *what a turn was*:

  - **A turn with no tool call is no longer rare.** The Instruct model acted on every turn but
    the last. A thinking model will happily spend a whole turn reasoning and ask for nothing,
    and the previous edition's answer to a turn like that — nudge it and try again — is an
    unbounded loop wearing a step budget as a disguise. Hence `idle_turns` and MAX_IDLE_TURNS
    below: the loop guard for thinking, next to the one for actions.

  - **The action guard must ignore reasoning.** `call_signature` deliberately hashes only the
    tool name and its arguments. A model that reasons its way to the same useless call by a
    fresh route each time is still stuck, and fresh reasoning must not buy a repeat another
    turn. This is a one-line decision that would be very easy to get backwards.

Everything else is the framework's, and the split is the point of the whole repo.

What the framework does here:

  - `ToolNode` runs the calls. Dispatch, ordering, unknown tool names, argument validation and
    error recovery — one invocation per turn, handed the batch that survived the guard.
  - `add_messages` makes the history append-only by construction, which is what keeps the
    prompt prefix byte-stable and the server's KV cache valid.
  - The other reducers on `AgentState` accumulate the counters, so `agent_node` returns deltas
    and never reads the old value.
  - Callbacks carry the trace, including the reasoning. No node below contains tracing code.
  - The checkpointer snapshots the state after every node, so a run can be resumed or
    inspected step by step.
  - `ChatOllama` separates the reasoning from the answer, so nothing here parses `<think>`.

What it does NOT do, and this is the part worth the workshop's time:

  - `handle_tool_errors` defaults to letting a tool's exception propagate and kill the run.
    The original guaranteed dispatch never raises. We opt back in, below.
  - **Malformed tool-call arguments are not dealt with for you.** Measured in
    `langchain_ollama/chat_models.py`: unparseable arguments are either kept leniently as the
    raw string (when they arrive inside a dict) or raise `OutputParserException` when they do
    not. The raise travels straight out of the model call and would end the run; `agent_node`
    below turns it back into a turn the model can learn from.

    The rule that makes this matter is the API's, not ours: every tool call the model made
    must get exactly one reply, matched by `tool_call_id`, and a request that leaves one
    unanswered is rejected on the NEXT turn — one step away from the code that caused it.
    Keeping that invariant is ours. See `guard_node`, where even a call the guard REFUSES to
    run still produces a message.
  - Either loop guard's POLICY, though the action guard's plumbing is now borrowed.
    `create_react_agent` wires `post_model_hook` by diffing the answered `tool_call_id`s
    against the calls the model made and dispatching only what is left, so answering a call is
    what refuses it. `guard_node` plus `route_after_guard` is that router, reproduced — the
    hook is a `create_react_agent` argument while `Send` is public — and there is no synthetic
    AIMessage anywhere in this file as a result. What the shape cost is measured in
    tests/test_hook_alternative.py: the verdict fold moved to `fold_node` to stay a single
    writer, and serialising a turn's calls became the run config's job. What no seam supplies
    is the claim that an identical call means the model is stuck. For the THINKING guard there
    is no seam at all — `after_model` could count idle turns, but the counter would live on the
    middleware rather than in `AgentState.idle_turns`, which is the same problem one step on.
  - The step budget, here. `recursion_limit` counts node executions, not model turns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from agentgraph.agent.state import AgentState, initial_state
from agentgraph.agent.trace import Tracer, TraceEvent, prompt_tokens_of, reasoning_of
from agentgraph.sandbox.base import ExecResult
from agentgraph.tasks.loader import Task
from agentgraph.tools.base import WorkspaceChanged

# This tool granularity needs run_tests + list_files + one read_file per implicated file +
# write_file + a verifying run_tests, which is already 8 steps for a three-file read.
MAX_STEPS = 10

# Three identical calls in a row is a stuck model, not slow progress.
MAX_GUARD_HITS = 3

# Consecutive turns of reasoning with no tool call before the run is abandoned. Two, not three:
# a thinking turn is the most expensive kind of turn there is — it is the one that generates
# hundreds of tokens — and a model that has been told twice to act and has not is not about to.
#
# One is too few. A model legitimately spends a turn planning after a surprising test failure,
# and cutting it off there would punish exactly the behaviour this edition exists to get.
MAX_IDLE_TURNS = 2

# Sent when the model replies without acting while the tests are still red. Two versions,
# because the useful thing to say depends on whether it thought first, and a nudge that
# misdiagnoses the turn is a nudge the model can reasonably ignore.
NUDGE = "The tests have not passed. Read the latest failure and write a fix."
NUDGE_AFTER_THINKING = (
    "You reasoned about the problem but did not call a tool, so nothing happened and the "
    "tests are still failing. Reasoning is not an action. Call a tool now — read the file "
    "the failure names, or call write_file with the fix you just described."
)

# Sent when the model's reply could not be READ at all — see `agent_node`. Addressed to the
# model rather than about it, because the model is the only thing that can fix it.
UNREADABLE_REPLY = (
    "Your last reply contained a tool call whose arguments were not valid JSON, so the reply "
    "could not be read and nothing ran. Send the call again, with the arguments as a single "
    "well-formed JSON object."
)


@dataclass(frozen=True)
class AgentResult:
    """Everything one run produced: the verdict plus what it cost to reach it."""

    task_id: str
    solved: bool
    steps_used: int
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    trace: tuple[TraceEvent, ...]
    peak_prompt_tokens: int = 0

    # How many of those steps arrived with reasoning attached. The headline number of this
    # edition: the previous one reported 0 of 7 and called it the open question.
    reasoning_turns: int = 0

    # Set only when the RUN ITSELF failed — the model server went away, the graph was wired
    # wrong — as opposed to the agent failing to fix the bug. `solved=False` cannot tell those
    # apart, and the difference is the difference between a score and a broken harness. It is a
    # plain string rather than the exception so it survives into the eval JSON, which drops the
    # trace. See eval/runner.crashed.
    error: str | None = None


def system_prompt(tools: Sequence[BaseTool]) -> str:
    """The standing instructions, rebuilt from whatever tools are actually registered.

    The tool names are derived from the tools themselves rather than hardcoded, so adding a
    tool updates the prompt and the schema together and they cannot drift apart.

    Most of this text is the result of watching the model fail: it read every file in the
    project, it emitted diffs instead of whole files, it declared victory without re-running
    the tests. Each instruction below is a countermeasure to an observed failure.

    Two of them are new, and both are countermeasures to REASONING rather than to acting. A
    thinking model has a failure mode the Instruct model could not have: deliberating instead
    of working. Telling it plainly that a turn must end in a tool call is cheaper than catching
    it afterwards with `idle_turns` — the guard is the backstop, this is the fix.
    """
    names = ", ".join(tool.name for tool in tools)
    return (
        "You are a Python bug-fixing agent working in a small project.\n"
        f"You have these tools: {names}.\n"
        "Think briefly about what to do next, then DO it in the same turn. Every turn must "
        "end in a tool call — reasoning alone changes nothing and wastes a step.\n"
        "Do not re-derive what you already know from earlier turns; the results of your "
        "previous tool calls are above and can be trusted.\n"
        "Work in this order: run the tests to see what fails, read the relevant file(s) "
        "before editing, then write the corrected file.\n"
        "Only read files the failure actually implicates — do not read every file "
        "list_files returns.\n"
        "When you call write_file you must supply the COMPLETE file contents, not a diff.\n"
        "Then run the tests again to confirm the fix worked — you are not finished until "
        "they pass. If they still fail, read the new failure and try again.\n"
        "Make the smallest change that fixes the failure. Do not rewrite unrelated code."
    )


def task_prompt(task: Task) -> str:
    """The task's own prompt, verbatim. A seam: the agent never invents task text."""
    return task.prompt


def is_done(state: AgentState) -> bool:
    """The agent is done when the tests actually pass — never because it says so.

    The highest-value idea in the whole design, and the only thing that can end a run early.
    Two failure modes it rules out: a model that claims a fix it never made, and a model that
    passed the tests once and then broke them again — see `tests_passed_after` for how the
    second one is kept out.

    Worth noting what reasoning did NOT change here. A thinking model produces something new
    and seductive: an articulate, step-by-step account of why the bug is fixed. It is still not
    evidence. This function does not read it.
    """
    return state["tests_passed"]


def tests_passed_after(replies: Sequence[AnyMessage], current: bool) -> bool:
    """Fold one turn's tool answers into the verdict, in the order they happened.

    Driven by artifact TYPE rather than tool name, so the rule is about what a tool did rather
    than what it is called: an `ExecResult` is evidence about the code as it stands, and a
    `WorkspaceChanged` means the code no longer stands as measured.

    Order matters within a single turn, which is why this is a fold and not a pair of ifs. A
    model that calls run_tests and then write_file in one message ends the turn red, and one
    that writes and then runs ends it with whatever the run said.

    Pass it THIS turn's replies, never the whole history. The checkpointer's serialiser
    round-trips these artifacts back as plain dicts, so the isinstance checks below hold only
    for messages this process just produced. The verdict itself is a bool in the state, which
    survives a checkpoint intact — which is the whole reason it is a bool in the state.
    """
    for message in replies:
        if not isinstance(message, ToolMessage):
            continue
        if isinstance(message.artifact, ExecResult):
            current = message.artifact.passed
        elif isinstance(message.artifact, WorkspaceChanged):
            # The tests may well still pass — but nothing has measured this code yet, and an
            # unmeasured guess is exactly what `is_done` exists to refuse.
            current = False
    return current


def call_signature(call: dict[str, Any]) -> str:
    """An identity for "the same call again": tool name plus its arguments.

    `sorted(...)` first so that {"a": 1, "b": 2} and {"b": 2, "a": 1} compare equal — key order
    in the model's JSON is not meaningful.

    What is NOT in this signature is the reasoning that produced the call, and that omission is
    load-bearing now that there is reasoning to omit. A thinking model rarely repeats itself
    word for word; it arrives at the same dead end by a slightly different argument each time.
    Include the reasoning here and every repeat looks novel, the guard never fires, and the run
    burns its whole budget re-reading one file. The guard watches what the model DID.
    """
    return f"{call['name']}::{sorted((call.get('args') or {}).items())!r}"


def requested_calls(message: AIMessage) -> list[tuple[dict[str, Any], str]]:
    """Every call the model made this turn, paired with its guard signature.

    Every call is assumed to carry an `id`, which is what a reply is paired with. Upstream types
    it `str | None`; ChatOllama always synthesises a uuid, so it is always there for us. A
    backend that omitted one would fail inside ToolMessage validation, and that is the right
    outcome — there is no correct reply to a call you cannot address.
    """
    return [(dict(call), call_signature(dict(call))) for call in message.tool_calls]


def acted(message: AIMessage) -> bool:  # noqa: ARG001
    """Did this turn ask for anything to happen? The distinction the thinking guard is built on.

    EXERCISE(stage-1): see exercises/stage_1/README.md
    """
    raise NotImplementedError("stage 1: did this turn ask for anything to happen?")


def guard_observation(name: str, hits: int) -> str:
    """The text sent back for a repeated call, escalating on the second repeat."""
    if hits == 1:
        return (
            f"You already called {name} with these exact arguments and got the result above. "
            "Try a different tool or different arguments."
        )
    # Second repeat: name the consequence. Being explicit that the run will be abandoned
    # measurably helps a small model break out of the pattern.
    return (
        f"You have now called {name} with identical arguments {hits + 1} times in a row and it "
        "was not executed. Call a different tool or use different arguments — read the file the "
        f"failure names, or call write_file with a fix. After {MAX_GUARD_HITS} repeats this run "
        "is abandoned."
    )


def completion_tokens_of(message: AIMessage) -> int:
    """Tokens the model generated this turn, or 0 if the server reported none.

    Reasoning is included in this number and cannot be separated out: Ollama reports one
    `output_tokens` covering the thinking and the answer together. So this is honestly the
    cost of the turn, and dishonestly a measure of the answer.
    """
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    return int(usage.get("output_tokens", 0))


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    tracer: Tracer,
    max_steps: int = MAX_STEPS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Assemble the agent. Returns a compiled graph you can `.invoke(state)`.

    Everything the nodes need that is not in the state — the model, the tools, the tracer — is
    captured here by closure. That is the graph equivalent of the original's "every tool was
    constructed with the workspace already bound": no node takes a path or a client as an
    argument, so no node can be pointed at the wrong workspace.
    """
    bound = llm.bind_tools(list(tools))
    # `handle_tool_errors=True` and not a custom message, for a reason worth knowing.
    # ToolNode's DEFAULT catches argument-validation errors but lets anything else — a genuine
    # exception inside a tool — propagate and kill the run, which breaks the original's "a tool
    # crash must not end the run" guarantee. So it has to be set.
    #
    # But passing a *string* here replaces the error text for every failure alike, and that
    # throws away the specific part: "path: Field required" becomes a generic apology, and the
    # model no longer knows which argument it forgot. `True` keeps ToolNode's own message,
    # which names the tool and the problem. Measured cost of getting this wrong: the model
    # retries blind.
    tool_node = ToolNode(list(tools), handle_tool_errors=True)

    def agent_node(state: AgentState) -> dict[str, Any]:
        """Ask the model what to do. The whole history is re-sent; models are stateless.

        No tracing and no timing in here: the tracer is a callback handler, so LangChain calls
        it around this `invoke` on its own. What is left is a state transition, which is all a
        node should be — plus one recovery, below, that has nowhere else to live.
        """
        try:
            reply = bound.invoke(state["messages"])
        except OutputParserException as error:
            # The model said something the client could not read. Measured in
            # `langchain_ollama`: unparseable tool-call arguments that did not arrive inside a
            # dict raise from here rather than being reported as a malformed call, so this is
            # the bad-JSON path on this client — there is no other one.
            #
            # Unhandled, it ends the run: `solve_task` propagates it and the task is recorded
            # as a CRASH, which is honest but wrong. Nothing is broken. The model produced one
            # malformed reply, exactly the way it sometimes produces a wrong file, and it can be
            # told and asked again. A small model gets this wrong often enough that the tier-2
            # option in the README would otherwise crash rather than score.
            #
            # There is no AIMessage to return: the reply never became one, which is the whole
            # problem. So nothing of the model's turn is appended — only the correction — and
            # `route_after_agent` recognises a non-AIMessage tail as this case. That is why it
            # cannot simply assert.
            tracer.note("llm", "assistant", f"unreadable reply — {type(error).__name__}")
            return {
                "messages": [HumanMessage(content=UNREADABLE_REPLY)],
                "step": 1,
                # A turn that could not be read changed nothing, so it counts as an idle turn
                # and MAX_IDLE_TURNS bounds the retries. Reusing the thinking guard rather than
                # adding a third counter: both are the same question — how many turns in a row
                # is this agent allowed to move nothing? The token counters are simply omitted;
                # their reducers leave them alone, and the server reported no usage to add.
                "idle_turns": state["idle_turns"] + 1,
            }

        assert isinstance(reply, AIMessage)
        thought = reasoning_of(reply)

        # Almost every value here is a DELTA, combined with what is already in the state by
        # that key's reducer: messages append, the counters add, the peak takes the maximum.
        return {
            # The message object itself is passed back untouched, which is what keeps the
            # prefix byte-stable for the server's KV cache — and keeps the reasoning attached,
            # since it rides along in `additional_kwargs`.
            "messages": [reply],
            "step": 1,
            "prompt_tokens": prompt_tokens_of(reply),
            "completion_tokens": completion_tokens_of(reply),
            "peak_prompt_tokens": prompt_tokens_of(reply),
            "reasoning_turns": 1 if thought else 0,
            # `idle_turns` belongs here, and it is not written like any of the keys above it.
            # See AgentState.idle_turns. Absent for now, so the counter never moves.
            #
            # EXERCISE(stage-1): see exercises/stage_1/README.md
        }

    def guard_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Refuse the calls that mean the model is stuck, by ANSWERING them.

        This is the framework's own shape for the job, in a hand-built graph. `route_after_guard`
        works out what still needs running by diffing the calls the model made against the
        `tool_call_id`s already answered — so answering a call here is what refuses it, and the
        surviving subset never has to be restated anywhere. `create_react_agent` wires its
        `post_model_hook` exactly this way. The hook itself is a `create_react_agent` argument
        rather than a `StateGraph` one, but `Send` is public, so the shape is ours to use.

        What no framework supplies is the policy: that a call identical to the last one means the
        model learned nothing from the result and re-running it would buy the same output for
        another step. Note that `call_signature` ignores the reasoning that led to the call —
        this edition's models talk themselves into the same useless call by a fresh route every
        time, and new reasoning must not buy a repeat another turn.

        The guard's own memory — the previous call's identity and the repeat count — lives in
        `AgentState`, which is what makes it survive a checkpoint and stay scoped to this run.
        agent/prebuilt.py keeps the same two values on a middleware instance instead, and its
        docstring records what that costs.
        """
        # The oracle guarantee. It used to be pinned on a single batched `ToolNode.invoke` from
        # inside this node; now the dispatch below is the GRAPH's, one task per call, so the only
        # place that can serialise it is the run config. Measured in
        # tests/test_hook_alternative.py: `max_concurrency=1` throttles Send fan-out too, so the
        # guarantee survives the change of shape — but it moved out of this file's reach.
        #
        # Which is exactly why it is checked here instead of documented and hoped for. Without
        # it, a turn calling write_file and run_tests together can have the tests measure the
        # file as it was BEFORE the write, and `fold_node` would then take that stale-but-green
        # ExecResult as the verdict — the precise "believe the tests, not the model" guarantee
        # this project is built on. A raise rather than an assert: `python -O` strips asserts,
        # and a false SOLVED is not the kind of thing to lose to an optimisation flag.
        if config.get("max_concurrency") != 1:
            raise RuntimeError(
                "the tool step fans out one task per call, so the run config must carry "
                "max_concurrency=1 or a turn's calls execute in parallel and run_tests can "
                "measure the workspace as it was before a write in the same turn; "
                f"got {config.get('max_concurrency')!r}"
            )

        message = state["messages"][-1]
        assert isinstance(message, AIMessage)

        replies: list[AnyMessage] = []
        signature = state["last_signature"]
        hits = state["guard_hits"]

        for call, current in requested_calls(message):
            name = str(call.get("name") or "unknown")

            if current == signature:
                hits += 1
                replies.append(
                    ToolMessage(
                        content=guard_observation(name, hits),
                        tool_call_id=call["id"],
                        name=name,
                    )
                )
                tracer.note("tool", name, f"guarded — identical call #{hits + 1} in a row")
                continue

            # Progress: reset the counter and remember this call as the new baseline. Note the
            # baseline advances for a call that is merely DISPATCHED, not one known to have
            # succeeded — deliberate: a call that ran and failed is still new information, and
            # repeating it verbatim is still the model going in circles.
            hits = 0
            signature = current

        return {"messages": replies, "last_signature": signature, "guard_hits": hits}

    def fold_node(state: AgentState) -> dict[str, Any]:
        """The single writer of `tests_passed`, and the reason it can stay reducer-free.

        This node exists because of the fan-out. With one task per call, several tool tasks land
        in the same superstep, and two writes to one reducer-free key in one step is something
        LangGraph refuses outright — `InvalidUpdateError`, measured in
        tests/test_hook_alternative.py. Folding here keeps one writer, which lets `tests_passed`
        stay a plain bool: state.py argues for that on its own merits, because a reducer is
        handed (current, incoming) and cannot tell "the suite went green" from "the workspace
        changed, so the verdict is void".

        It also carries the invariant the tool step has to uphold. Every call the model made must
        get exactly one reply, matched by `tool_call_id`; leave one unanswered and the API rejects
        the NEXT request, a turn away from the cause. Checked here because here is the first point
        where the whole turn is visible — the guard's refusals and the tools' answers together.
        """
        messages = state["messages"]
        # THIS turn's replies, and no more: everything appended after the last AIMessage. The
        # fold recognises artifacts by TYPE, and a checkpoint round-trip hands them back as plain
        # dicts — so folding the whole history would quietly stop recognising anything at all.
        last_ai = next(
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], AIMessage)
        )
        asked = messages[last_ai]
        assert isinstance(asked, AIMessage)
        turn = messages[last_ai + 1 :]

        requested = [call["id"] for call in asked.tool_calls]
        answered = [m.tool_call_id for m in turn if isinstance(m, ToolMessage)]
        if len(answered) != len(requested):
            raise RuntimeError(
                f"the tool step answered {len(answered)} of {len(requested)} tool calls; "
                "every call the model made must get exactly one reply"
            )

        return {"tests_passed": tests_passed_after(turn, state["tests_passed"])}

    def nudge_node(state: AgentState) -> dict[str, Any]:
        """A reply that acted on nothing is not a stop condition — say so and go again.

        Right now every such turn gets the same correction, and there are two to choose from.

        EXERCISE(stage-1): see exercises/stage_1/README.md
        """
        message = state["messages"][-1]
        assert isinstance(message, AIMessage)
        return {"messages": [HumanMessage(content=NUDGE)]}

    def route_after_agent(state: AgentState) -> str:
        """Where to go after a model turn. The only place a run can end successfully."""
        message = state["messages"][-1]

        # Not an AIMessage means `agent_node` could not read the model's reply and appended a
        # correction instead of a turn. Go straight back to the model — there is nothing to
        # execute and nothing to finish on — but only while both budgets still allow it, so a
        # model that cannot emit valid JSON at all cannot spin here.
        if not isinstance(message, AIMessage):
            if state["step"] >= max_steps:
                return END
            if state["idle_turns"] >= MAX_IDLE_TURNS:
                tracer.note(
                    "llm",
                    "assistant",
                    f"abandoned — {state['idle_turns']} consecutive turns that changed nothing, "
                    "the last one unreadable",
                )
                return END
            return "agent"

        # Tool calls never end the run on their own — always execute them and loop back, so
        # the model can read the results. Note this skips the `is_done` check on purpose: the
        # check belongs on a turn where the model had nothing more to do.
        if acted(message):
            return "guard"

        # No action: the model spent a turn and asked for nothing. Four answers from here, and
        # their order is as much of the decision as the answers are.
        #
        # EXERCISE(stage-1): see exercises/stage_1/README.md
        raise NotImplementedError("stage 1: the model asked for nothing — now what?")

    def route_after_guard(state: AgentState) -> str | list[Send]:
        """Dispatch whatever the guard let through. The framework's router, reproduced.

        `pending` is the calls the model made minus the ones already answered — which is the
        whole mechanism behind refusing by answering. Each one is `Send`-ed as its own task, so
        no message anywhere has to state the surviving subset.

        The diff is scoped to THIS TURN, and that is deliberately stricter than the framework's
        own version of this router, which collects answered ids from the whole message history
        and so quietly assumes `tool_call_id`s are globally unique. Real ones are — ChatOllama
        synthesises a uuid per call. But the ids are the MODEL's to choose, and a model that
        reuses one would have its second call silently treated as already answered and never
        run. Measured with llm/fake.py, whose scripted calls all carry `call_1`: against the
        whole history, turn two dispatched nothing and the fold then reported an unanswered
        call. A turn is the only window in which the pairing is guaranteed to mean anything.

        When the guard refused everything, there is nothing to dispatch and no tool will run,
        but the turn still has to be folded and routed — so it goes straight to `fold`.
        """
        messages = state["messages"]
        last_ai = next(
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], AIMessage)
        )
        asked = messages[last_ai]
        assert isinstance(asked, AIMessage)
        answered = {
            m.tool_call_id for m in messages[last_ai + 1 :] if isinstance(m, ToolMessage)
        }
        pending = [call for call in asked.tool_calls if call["id"] not in answered]
        if not pending:
            return "fold"
        # `type="tool_call"` selects ToolNode's per-call Send payload: it hydrates the state it
        # needs from the graph's channels instead of being handed an inlined snapshot.
        return [Send("tools", [dict(call, type="tool_call")]) for call in pending]

    def route_after_tools(state: AgentState) -> str:
        """Stop if the model is stuck or out of budget; otherwise take another turn.

        Note what is NOT here: `is_done`. A turn whose tools just went green does not end the
        run — the model gets one more turn, and `route_after_agent` ends it there. That is a
        deliberate choice rather than an oversight, and it is not free: it costs one model turn
        per solved task, which on a thinking model is the most expensive turn in the run.

        What it buys is the closing statement, and in this edition that is worth more than it
        was in the last one. That final turn is where the model explains the fix it made, with
        the reasoning channel attached — so a solved run ends with an account of the bug in the
        model's own words, which is the artifact the whole workshop is building toward. Ending
        on the green test result would stop collecting it. It also keeps every trace the same
        shape, ending on an `llm` line whether the run succeeded or not, which is what makes
        two of them comparable side by side.

        The extra turn is not a soundness hole: if the model spends it on a write, the fold
        clears the verdict and the run correctly carries on.
        """
        if state["guard_hits"] >= MAX_GUARD_HITS:
            return END
        if state["step"] >= max_steps:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("guard", guard_node)
    # ToolNode is added as a NODE now, rather than invoked by hand from inside one. Dispatch,
    # ordering, unknown tool names, argument validation and the `handle_tool_errors` recovery
    # are its job, and the graph drives it one call at a time through the Send above.
    graph.add_node("tools", tool_node)
    graph.add_node("fold", fold_node)
    graph.add_node("nudge", nudge_node)
    graph.add_edge(START, "agent")
    # "agent" is a destination of its own router: the unreadable-reply path retries the model
    # directly, with no tool call to execute and no correction node to pass through.
    graph.add_conditional_edges("agent", route_after_agent, ["agent", "guard", "nudge", END])
    graph.add_conditional_edges("guard", route_after_guard, ["tools", "fold"])
    graph.add_edge("tools", "fold")
    graph.add_conditional_edges("fold", route_after_tools, ["agent", END])
    graph.add_edge("nudge", "agent")

    # With a checkpointer the graph writes a snapshot of the state after every node, keyed by
    # the `thread_id` in the run config. That buys resumption and, in a debugger, time travel:
    # `get_state_history(config)` hands back every step this run went through.
    #
    # Worth knowing that this only became honest once the verdict moved into the state. While
    # `tests_passed` lived on the run_tests tool, a resumed run rebuilt that tool empty and a
    # solved task came back unsolved — the graph was checkpointable and the agent was not.
    return graph.compile(checkpointer=checkpointer)


def run_agent(
    task: Task,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    max_steps: int = MAX_STEPS,
    tracer: Tracer | None = None,
) -> AgentResult:
    """Run the agent until the tests pass, the step budget runs out, or it gets stuck.

    Note there is no `work_dir` parameter: every tool was constructed with the workspace
    already bound (see runner.py), so nothing here touches a path. And no `run_tests`
    parameter either — the verdict arrives in the state as the tools answer, so nothing here
    needs a handle on the oracle itself.
    """
    # `tracer or Tracer()` rather than a default argument of `Tracer()`: a mutable default is
    # evaluated once at function definition and would be shared by every call.
    tracer = tracer or Tracer()
    # One saver per run, so a second run of the same task cannot resume the first one's thread.
    app = build_graph(llm, tools, tracer, max_steps=max_steps, checkpointer=InMemorySaver())

    started = time.time()
    # `recursion_limit` is LangGraph's own backstop and counts SUPERSTEPS, not model
    # turns: a turn is agent + tools, and sometimes a nudge as well. Set generously, because
    # the budget that actually matters is `max_steps`, enforced in the routers above. Hitting
    # this limit raises, which is the correct behaviour for "the graph is wired wrong".
    final: AgentState = app.invoke(
        initial_state(system_prompt(tools), task_prompt(task)),
        config={
            # The tracer is handed to the framework here, once, and LangChain calls it around
            # every model and tool invocation inside the graph — including the ones ToolNode
            # makes on our behalf. This is why no node contains tracing code.
            "callbacks": [tracer],
            # Not a performance knob. The tool step fans out one task per call, and these
            # tools are not independent: run_tests measures what write_file just wrote. One
            # worker keeps a turn's calls in the order the model asked for them, which is what
            # `guard_node` refuses to run without. See tests/test_hook_alternative.py.
            "max_concurrency": 1,
            "recursion_limit": max_steps * 5 + 10,
            # Which conversation this is. One task, one thread — the checkpointer files every
            # snapshot under it, and resuming means invoking again with the same id.
            "configurable": {"thread_id": task.task_id},
        },
    )

    return AgentResult(
        task_id=task.task_id,
        # Read from the final state, not from how the graph exited: ending on MAX_GUARD_HITS
        # or running out of steps must not be mistaken for success.
        solved=is_done(final),
        steps_used=final["step"],
        prompt_tokens=final["prompt_tokens"],
        completion_tokens=final["completion_tokens"],
        duration_s=round(time.time() - started, 2),
        trace=tuple(tracer.events),
        peak_prompt_tokens=final["peak_prompt_tokens"],
        reasoning_turns=final["reasoning_turns"],
    )
