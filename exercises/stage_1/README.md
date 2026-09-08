# Stage 1 — A turn that changed nothing

Open `src/agentgraph/agent/graph.py` and find the three `EXERCISE(stage-1)` markers. Together they
are one decision: what happens when the model spends a turn and asks for nothing.

The Instruct model in the previous edition acted on every turn but the last, so a turn like that
was the end of the run or a rare mistake. A thinking model does it on purpose. It will reason for
three hundred tokens, conclude that the fix is obvious, and call nothing — and the previous
edition's answer to a turn like that was to nudge it and go again, which is an unbounded loop
wearing a step budget as a disguise.

## What to write

**1. `acted(message)`** — one line. Did this turn ask for anything to happen? A turn that called
a tool moved something; a turn that only talked did not.

`message.tool_calls` is what the model asked for this turn, and it is the only thing this function
may look at. A turn that reasoned for three hundred tokens and called a tool acted; a turn that
reasoned for three hundred tokens and called nothing did not — ReAct means thinking and acting in
the *same* turn, so the reasoning is never what makes the difference. (A reply the client could not
parse never becomes a message at all, so it never reaches here; `agent_node` handles that one.)

**2. `idle_turns` in `agent_node`'s returned dict.** Zero if the turn acted, otherwise one more
than the state already holds.

This is the only key in `AgentState` written as an absolute value rather than a delta, and that
is the part worth slowing down for. Every other counter — `step`, the token totals,
`reasoning_turns` — is `Annotated[int, operator.add]`: the node returns `1` and the reducer adds
it to what is there. Try that here and the counter can only ever grow, because **a reducer cannot
express a reset**. It is handed `(current, incoming)` and nothing else, so it cannot tell "one
more idle turn" from "that turn acted, start again". `agent_node` can tell, because it is holding
the reply. That makes it the single writer of this key, and the reason it returns the whole value.

**3. `nudge_node`'s choice of nudge.** `NUDGE` and `NUDGE_AFTER_THINKING` are both written for
you, and `reasoning_of(message)` tells you which turn you are looking at. A model that said
nothing needs pointing at the failure; a model that reasoned its way to a conclusion and then
stopped needs telling that a conclusion is not a change. A nudge that misdiagnoses the turn is a
nudge the model can reasonably ignore.

## The traps

**Read the router you were given.** `route_after_agent`'s no-action tail is written for you, and
the order of its four answers is the whole of it: the check on the tests comes FIRST. A solved
run's closing turn is a turn with no tool call — that is what a model does when it has nothing
left to do — and any guard that reads it before `is_done` reports a successful run as a stalled
one. Your `acted` and your `idle_turns` are what that ordering consumes, so it is worth reading
before you write either.

**`MAX_IDLE_TURNS` is 2, not 1.** A model legitimately spends a turn planning after a surprising
failure, and cutting it off there punishes exactly the behaviour this edition exists to get. Two
turns of being told to act and not acting is a different thing.

**The reset is not decoration.** Without it, thinking turns spread across a long run accumulate
into a stall that never happened: think, act, think, act — and the fourth turn is abandoned. Half
the tests in this stage exist to catch that one mistake.

**A turn the client could not read is also a turn that changed nothing.** That path is already
written, at the top of `agent_node` and again at the top of the router, and it increments this
same counter rather than a third one of its own. Both are the same question: how many turns in a
row may this agent move nothing?

## Why this is not the framework's job

LangGraph will run your nodes, thread the state through the reducers, checkpoint every step and
route on whatever your router returns. It has no opinion on whether a turn was worth taking.
`create_agent` and its middleware do not either — `agent/prebuilt.py` builds the same agent out
of the prebuilt pieces, and this counter is one of the things it cannot keep. Its `after_model`
hook could count idle turns, but the tally would live on the middleware *instance* rather than in
the run's state: not checkpointed, and leaking into the next run. So the prebuilt edition treats a
thinking turn exactly like silence and nudges it until the budget is gone. Read its docstring —
what the framework owns the state for, you cannot keep a policy in.

Which is the shape of the whole workshop. The framework absorbed the plumbing and left you the
judgement — and "a thinking model that deliberates instead of working is stuck" is judgement
about a 12B model on a three-file project, not something a graph library could know.

## Run it

    uv run python -m unittest exercises.stage_1.test_stage_1 -v

Then the real thing, against a real bug, with a real thinking model:

    uv run agentgraph solve tasks/workshop/01-shopcart --verbose

Watch the `thinks` lines. Some runs plan for a turn before touching anything — that is the turn
`MAX_IDLE_TURNS` of 1 would have killed.

## Finished early?

Two questions, both real.

`call_signature` hashes the tool name and its arguments and deliberately **not** the reasoning
that led to them. Work out what breaks if you add the reasoning — the model rarely repeats itself
word for word, so what happens to the action guard when every repeat arrives dressed in a fresh
argument?

Then `route_after_tools`, which stops on the two failure exits and deliberately does **not**
check `is_done`. It costs one model turn per solved task, and on a thinking model that is the
most expensive turn in the run. Its docstring says what the turn buys. Do you agree with the
trade?
