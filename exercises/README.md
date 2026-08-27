# Exercises

One stage. It edits `src/agentgraph/agent/graph.py` — the agent itself — and the tests run
**without a model**, so you can finish it offline and in any setup tier.

| Stage | You write | Test |
|---|---|---|
| 1 | the thinking guard — what a turn that changed nothing costs, and when a run of them ends | `uv run python -m unittest exercises.stage_1.test_stage_1 -v` |

One stage rather than two, because this edition only forced one new decision.

The sibling `agentfix-langchain` has two:
`route_after_agent` — where a run may end and on whose word — and the loop guard that catches a
model repeating itself. Both are written for you here, and reading them is the fastest way in:
the first is the rule that the tests decide when a run is over, the second is the rule that a
model which has stopped learning must be stopped. Neither changed when the model started
reasoning. What changed is that a turn can now cost three hundred tokens and move nothing, and
neither of those two guards can see a turn like that — one watches the verdict, the other
watches actions, and a turn that only thinks produces neither.

That hole is this stage. The previous workshop's stage 2 ends by pointing straight at it.

## Running the tests

    uv run python -m unittest exercises.stage_1.test_stage_1 -v      # this stage
    uv run python -m unittest discover -s exercises -t . -v          # same thing, by discovery

The repo's own suite is a superset and will also go green as you go — `tests/test_reasoning.py`
in particular, which is where this edition's behaviour is pinned:

    uv run python -m unittest discover -s tests -t .

On a fresh `main` clone the stage fails and `uv run agentgraph solve ...` does not work. That is
the intended starting point, not a broken checkout.

## Stuck?

Jump ahead without falling behind the room:

    git checkout stage-1-solution     # the stage, done

Or read the answer without moving your working tree:

    git diff main stage-1-solution -- src/agentgraph/agent/graph.py

To get back to your own work: `git checkout main`.
