"""The same agent, assembled out of the framework's prebuilt agent instead of a graph.

Not the path `agentgraph solve` takes. This file exists to answer one question honestly — how
much of agent/graph.py is *this project*, and how much is a tool-calling loop that the
framework will build for you? — and the answer moved between LangGraph 0.x and LangChain 1.x,
which is why it is worth having in the repo rather than in a blog post.

`create_agent` gives you the loop: model, tools, routing, and the ToolNode underneath. Its
middleware then gives you a hook at every point our graph hand-wired:

    ModelCallLimitMiddleware(run_limit=N)   the step budget, counting MODEL CALLS
    AgentMiddleware.wrap_tool_call          intercept a call before the tool runs
    AgentMiddleware.after_model + jump_to   send the model back for another turn

That first one is worth pausing on. The README used to say the framework cannot give you a
step budget because `recursion_limit` counts node executions rather than model turns. On
LangChain 1.x that is no longer true: `ModelCallLimitMiddleware` counts exactly the thing our
`AgentState.step` counts. The other two are seams rather than answers — the framework will run
your code at the right moment, but "an identical call means the model is stuck" and "prose
while the suite is red is not a stop condition" are still yours to write. That is the honest
line, and it moved: what used to be missing plumbing is now missing *policy*.

On reasoning, the framework comes out well, and it is worth saying so plainly. Nothing in
this file mentions thinking, and it did not have to: `reasoning=True` is set on the CLIENT
(llm/client.py), so `create_agent` inherits a reasoning model for free and the middleware below
never sees a `<think>` tag. Reasoning is a property of the model you hand in, not of the loop —
which is the correct place for it, and both editions get it right.

What is still genuinely absent, and the reason graph.py does not simply become this file:

  - **The verdict cannot survive a checkpoint.** `create_agent` carries its own state schema, so
    there is nowhere to keep a `tests_passed` bool — it has to be recomputed by folding the tool
    artifacts out of the message history. The checkpointer's serialiser round-trips those
    artifacts back as plain dicts, so after a resume every isinstance check fails and the fold
    returns its `False` seed. Measured: live run `prebuilt_solved=True`, same thread resumed
    `False`, and `VerifiedStop` then nudges a GREEN suite until the budget runs out. This is the
    exact bug the opening paragraph credits agent/graph.py with fixing, and it is back, because
    the state is the framework's rather than ours. `build_prebuilt_agent` therefore refuses a
    checkpointer.
  - The guard never abandons a stuck model. It answers a repeated call but cannot end the run,
    so nine identical calls cost nine model turns; agent/graph.py gives up after MAX_GUARD_HITS.
  - **There is no thinking guard, and no seam that fits one well.** A model that reasons and
    calls nothing is the characteristic failure of a thinking model, and `VerifiedStop` below
    treats it exactly like silence: nudge, and go again, until the budget is gone. `after_model`
    could count those turns, but the counter would live on the middleware instance and inherit
    LoopGuard's two problems below — not checkpointed, and leaking into the next run. In
    agent/graph.py it is `AgentState.idle_turns`, which is scoped to the run because the state
    is. This is the same lesson as the verdict, one level along: what the framework owns the
    state for, you cannot keep a policy in.
  - Middleware state is per-instance rather than in the state, with two consequences — see
    LoopGuard below.
  - There is no graph to read. For a workshop whose subject is what an agent actually does,
    three explicit nodes and two routers beat a constructor call — which is the whole reason
    this is the comparison and not the implementation.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    hook_config,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from agentgraph.agent.graph import (
    MAX_STEPS,
    NUDGE,
    call_signature,
    guard_observation,
    system_prompt,
    tests_passed_after,
)


class LoopGuard(AgentMiddleware):
    """Refuse a tool call identical to the previous one, and say so instead of running it.

    `wrap_tool_call` is handed the call and a `handler` that would execute it. Returning a
    ToolMessage without calling the handler is the whole guard: the model gets an answer to the
    call it made, and the tool never ran. The framework supplies the seam; the policy — that an
    identical call means the model learned nothing and is stuck — is still ours.

    The counters live on `self`, which is the compromise this shape forces, and it cuts both
    ways. agent/graph.py keeps the same two values in `AgentState`, where a checkpoint preserves
    them across a resume — and where they are scoped to the run rather than to the object.

    Here they are neither. They are not checkpointed, so a resumed run forgets the model was
    repeating itself; and they LEAK FORWARD, so invoking one compiled agent twice refuses the
    second run's first tool call as a repeat of the first run's last one. Measured: run 2's
    opening `list_files` came back "You already called list_files with these exact arguments".
    `build_prebuilt_agent` is safe to call again (it constructs a fresh guard); the agent it
    returns is not safe to reuse across tasks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_signature: str | None = None
        self.hits = 0

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        call = dict(request.tool_call)
        current = call_signature(call)
        if current == self.last_signature:
            self.hits += 1
            return ToolMessage(
                content=guard_observation(str(call["name"]), self.hits),
                tool_call_id=str(call["id"]),
                name=str(call["name"]),
            )
        self.hits = 0
        self.last_signature = current
        return handler(request)


class VerifiedStop(AgentMiddleware):
    """A prose reply while the tests are red is not a stop condition.

    `jump_to="model"` is what makes this expressible at all: without it the agent ends the
    moment the model stops calling tools, whatever the suite says.

    The `@hook_config` decorator declares that this hook may send the graph back to the model,
    and the framework reads `can_jump_to` off the method to decide whether to WIRE that edge at
    all. Get it wrong and `{"jump_to": "model"}` is ignored, the run ends on the first prose
    reply, and the only symptom is an agent that believes itself.

    Note what this does NOT distinguish, and agent/graph.py does: whether the model reasoned
    before it stopped. Both editions send a nudge; only the graph picks between NUDGE and
    NUDGE_AFTER_THINKING, and only the graph counts the turn toward giving up. Against a
    thinking model those two omissions compound — the most expensive kind of turn, repeated
    until the budget runs out, answered each time with advice aimed at the wrong failure.

    Measured, because the first version of this comment was wrong. Whether the decorator is
    required depends on where this middleware lands in the list (langchain 1.3.16,
    langchain-core 1.6.0):

        VerifiedStop IS the after_model loop-exit hook   decorated=5  undecorated=5
        VerifiedStop is NOT the loop-exit hook           decorated=5  undecorated=1

    The loop-exit hook's edge consults `jump_to` unconditionally, so in `build_prebuilt_agent`'s
    own ordering the decorator happens to be inert. Move one middleware ahead of it and the
    undecorated version silently ends after a single model call. Which is the argument for the
    decorator rather than against it: it is what makes the behaviour independent of position,
    and position is not something you want a stop condition to depend on.
    """

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        if tests_passed_after(state["messages"], False):
            return None
        return {"messages": [HumanMessage(content=NUDGE)], "jump_to": "model"}


def build_prebuilt_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    max_steps: int = MAX_STEPS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """The agent as `create_agent` plus three pieces of middleware. Compare with build_graph.

    `checkpointer` exists only to be refused, and that is the point rather than laziness: this
    agent's verdict is recomputed from message artifacts that a checkpoint round-trip destroys,
    so a resumed run would report a solved task unsolved and nudge a passing suite until it ran
    out of budget. Accepting the argument and quietly doing that is how the original version of
    this bug survived in agent/graph.py for as long as it did. Raising names the limitation at
    the only moment anyone would hit it.
    """
    if checkpointer is not None:
        raise NotImplementedError(
            "the prebuilt agent has nowhere to keep the verdict, so it cannot be resumed: "
            "the tool artifacts it recomputes from come back from a checkpoint as plain dicts. "
            "Use agent.graph.build_graph, which keeps tests_passed in AgentState."
        )
    # ORDER IS LOAD-BEARING, and getting it wrong costs you the step budget silently.
    #
    # Routing takes whatever `jump_to` it finds in the state, and `after_model` hooks run in
    # reverse list order. Put the limit FIRST and VerifiedStop's `jump_to: "model"` is written
    # afterwards and wins every time: measured with run_limit=3, the model was called 20 times
    # and stopped only because the scripted fake ran out of replies. Put the limit LAST and its
    # decision to end is the one routing sees — 3 calls, as asked.
    #
    # Nothing warns you. A budget that quietly does not apply is worse than no budget, and it
    # is a bug that `if step >= max_steps: break` in a plain loop cannot have.
    #
    # `list[Any]` because the shipped middleware and our two are generic over different state
    # types, and mypy will not unify them.
    middleware: list[Any] = [
        LoopGuard(),
        VerifiedStop(),
        ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="end"),
    ]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt(tools),
        middleware=middleware,
    )


def prebuilt_solved(final_state: dict[str, Any]) -> bool:
    """Did the prebuilt agent actually fix the bug? Same rule, read from its messages.

    `create_agent` carries its own state schema, so there is no `tests_passed` key to read —
    the verdict has to be recomputed by folding the tool artifacts every time it is wanted.
    Cheap here, and a fair illustration of what the custom state in agent/graph.py buys.
    """
    return tests_passed_after(final_state["messages"], False)
