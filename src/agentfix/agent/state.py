"""The agent's state: what travels between the graph's nodes.

In the no-framework edition all of this lived in local variables inside one `for` loop. Making
it an explicit, typed structure is the single biggest change the framework asks of you, and it
buys two things: any node can read the whole state without arguments being threaded through,
and the state can be checkpointed, so a run can be resumed or inspected mid-flight.

It costs something too. In a `for` loop, `guard_hits += 1` is obviously correct. Here a node
returns a *partial* state which LangGraph merges, so "add one to the counter" is not something
a node can simply do — and reading the old value to return a new one is a bug waiting for the
day two nodes update the same key.

The answer is a reducer: a function that says how an update combines with what is already
there. `add_messages` is the one the framework ships; the numeric keys below use `operator.add`
and `max`. A node then returns a DELTA and never reads the old value at all.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def keep_larger(current: int, incoming: int) -> int:
    """The reducer for a high-water mark: the bigger of the two wins.

    Written out rather than passing the builtin `max`, and not by choice. LangGraph inspects a
    reducer's signature to decide it is a two-argument combiner, and `inspect.signature` raises
    on a C builtin — `Annotated[int, max]` fails at StateGraph construction with "no signature
    found for builtin max". A plain Python function has a signature, so this works.
    """
    return current if current >= incoming else incoming


class AgentState(TypedDict):
    """Everything the graph carries between nodes."""

    # `add_messages` is a *reducer*: a node returning {"messages": [x]} appends rather than
    # replaces. This is what keeps the history append-only, which is what keeps the prefix
    # byte-stable, which is what keeps the server's KV cache valid across turns.
    messages: Annotated[list[AnyMessage], add_messages]

    # Which model turn we are on, 1-based. This is the step budget from the original loop:
    # the framework's own `recursion_limit` counts node executions, which is a different and
    # less meaningful number — two nodes run per turn, plus a nudge sometimes.
    step: Annotated[int, operator.add]

    # Cost is reported, not hidden. The total is the bill; the peak is how close a single
    # prompt came to the context window. They answer different questions, and they reduce
    # differently: a bill sums, a high-water mark takes the larger of the two.
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    peak_prompt_tokens: Annotated[int, keep_larger]

    # How many turns arrived carrying reasoning. New in this edition, and the number the
    # previous workshop could not produce: its README could only report "0 of 7 turns had
    # reasoning" and leave closing that gap as the sequel. This is that gap, measured.
    #
    # It counts TURNS, not tokens, and that is a deliberate limit rather than an oversight.
    # Ollama reports one `output_tokens` figure covering the reasoning and the answer together,
    # so a reasoning-token count here would be a number we made up. A turn either reasoned or
    # it did not, and that we can actually observe.
    reasoning_turns: Annotated[int, operator.add]

    # Loop-guard state for ACTIONS: the previous call's identity, and how many times it has
    # repeated. Note that a call's signature ignores the reasoning that led to it — see
    # graph.call_signature. That is on purpose: a model which talks itself into the same
    # useless call by a fresh route each time is still stuck, and dressing a repeat in new
    # reasoning must not buy it another turn.
    last_signature: str | None
    guard_hits: int

    # Loop-guard state for THINKING: consecutive turns that requested no tool at all.
    #
    # This is the hole the previous workshop pointed at and did not fill. Its guard watched
    # actions, so a model that thinks and thinks and never acts walked straight past it and
    # spent the entire step budget in silence. A thinking model does this often enough to be
    # the characteristic failure of this edition.
    #
    # The one key here with no reducer, and a real reason for it. Every other counter above
    # only ever grows, which is what makes `operator.add` and a returned delta enough. This one
    # has to RESET — a turn that acted puts it back to zero — and a reducer is handed only
    # (current, incoming), so it cannot tell "add one more idle turn" from "that turn acted,
    # start again". `agent_node` is the single writer and knows which of the two just happened,
    # so it returns the absolute value.
    idle_turns: int

    # The agent's whole verdict, and the only thing that can end a run successfully. It lives
    # here rather than on the run_tests tool for a reason worth knowing: state is what gets
    # checkpointed. With the verdict hidden on a tool object, resuming a run rebuilt the tool
    # with no result and a solved task came back unsolved. Kept honest by two tool artifacts —
    # an ExecResult sets it, a WorkspaceChanged clears it. See graph.tests_passed_after.
    tests_passed: bool


def initial_state(system: str, task: str) -> AgentState:
    """The only two messages that exist before the model has done anything."""
    from langchain_core.messages import HumanMessage, SystemMessage

    return AgentState(
        messages=[SystemMessage(content=system), HumanMessage(content=task)],
        step=0,
        prompt_tokens=0,
        completion_tokens=0,
        peak_prompt_tokens=0,
        reasoning_turns=0,
        last_signature=None,
        guard_hits=0,
        idle_turns=0,
        tests_passed=False,
    )
