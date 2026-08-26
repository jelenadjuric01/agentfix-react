"""Measurement: run the agent over many tasks and report what happened.

The distinction worth understanding here is eval vs tests. The test suite asks "does the code
do what I wrote it to do?" and answers deterministically with a scripted fake model. Eval asks
"is the agent any good?" and needs a real model, so it is slow, costs tokens, and gives a
different answer each run. Both are necessary; neither substitutes for the other.

pass@1 is the headline: of N tasks, how many did the agent fix on its first and only attempt.
Steps, tokens and peak context are reported alongside it because an agent that solves a task in
10 steps and 40k tokens is not the same product as one that does it in 4 and 8k.

This edition adds a `thinks` column — how many of those steps arrived with reasoning attached.
It is here to be compared against the other columns rather than maximised. Reasoning is not
free: it is generated tokens, it is re-sent on every later turn (see llm/client.py), and it
therefore shows up in `tokens`, in `peak ctx` and in `seconds`. A run with more thinking and
the same pass@1 bought nothing.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agentgraph.agent.graph import MAX_STEPS, AgentResult
from agentgraph.agent.trace import TraceEvent
from agentgraph.config import REPO_ROOT
from agentgraph.eval.humanevalfix import load_vendored_rows, write_task_dir
from agentgraph.runner import solve_task

RESULTS_DIR = REPO_ROOT / "results"
WORKSHOP_TASKS_DIR = REPO_ROOT / "tasks" / "workshop"


@dataclass(frozen=True)
class EvalReport:
    """The results of one suite run, plus the derived numbers worth quoting."""

    suite: str
    results: tuple[AgentResult, ...]

    @property
    def pass_at_1(self) -> float:
        """Fraction of tasks solved. One attempt each — no retries, no best-of-n."""
        if not self.results:
            return 0.0  # an empty suite scores zero, rather than dividing by zero
        return sum(1 for result in self.results if result.solved) / len(self.results)

    @property
    def crashes(self) -> tuple[AgentResult, ...]:
        """Rows where the run failed, as distinct from the agent failing to fix the bug."""
        return tuple(result for result in self.results if result.error)

    @property
    def reasoning_turns(self) -> int:
        """Turns that carried reasoning, across the whole suite."""
        return sum(result.reasoning_turns for result in self.results)

    @property
    def steps_used(self) -> int:
        """Model turns across the whole suite — the denominator for `reasoning_turns`."""
        return sum(result.steps_used for result in self.results)

    @property
    def peak_prompt_tokens(self) -> int:
        """Largest single prompt across the whole suite — compare it to the context window."""
        # `default=0` because max() of an empty sequence raises.
        return max((result.peak_prompt_tokens for result in self.results), default=0)

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "pass_at_1": self.pass_at_1,
            "peak_prompt_tokens": self.peak_prompt_tokens,
            "reasoning_turns": self.reasoning_turns,
            "steps_used": self.steps_used,
            "results": [
                # The trace is dropped: it is by far the largest field, and a full trace per
                # task would bury the numbers this file exists to report.
                {k: v for k, v in asdict(result).items() if k != "trace"}
                for result in self.results
            ],
        }

    def format_table(self) -> str:
        """A fixed-width table for the terminal. `:<24` means left-align in 24 columns."""
        header = (
            f"{'task':<24} {'solved':<8} {'steps':<7} {'thinks':<8} {'tokens':<9} "
            f"{'peak ctx':<10} {'seconds':<8}"
        )
        rows = [
            f"{r.task_id:<24} {('CRASH' if r.error else str(r.solved)):<8} {r.steps_used:<7} "
            f"{f'{r.reasoning_turns}/{r.steps_used}':<8} "
            f"{r.prompt_tokens + r.completion_tokens:<9} {r.peak_prompt_tokens:<10} "
            f"{r.duration_s:<8}"
            for r in self.results
        ]
        summary = (
            f"\npass@1 = {self.pass_at_1:.2f}  ({len(self.results)} task(s))"
            f"  peak prompt = {self.peak_prompt_tokens} tok"
            f"  reasoning on {self.reasoning_turns}/{self.steps_used} turns"
        )
        if self.crashes:
            # Said next to the number, because that is where it will be read from.
            summary += (
                f"\n{len(self.crashes)} of {len(self.results)} task(s) CRASHED — "
                "that part of this number measures the harness, not the agent"
            )
        return "\n".join([header, "-" * len(header), *rows]) + summary


def crashed(task_dir: Path, error: Exception) -> AgentResult:
    """An unsolved result standing in for a task whose run raised.

    Recorded rather than propagated, because the alternative is worse than it looks: a suite
    of twenty HumanEvalFix tasks is ten minutes of a real model, and one exception on task
    seventeen would otherwise discard all sixteen finished results along with it. A crash is a
    fact about a task, so it belongs in the row for that task.
    """
    return AgentResult(
        task_id=task_dir.name,
        solved=False,
        steps_used=0,
        prompt_tokens=0,
        completion_tokens=0,
        duration_s=0.0,
        trace=(TraceEvent(0, "error", type(error).__name__, str(error), 0, 0.0),),
        # Duplicated out of the trace on purpose: `to_json` drops the trace, so without this
        # field the report said `solved: false` and kept no record of the fact that no model was
        # ever reached. A row that cannot explain itself becomes a publishable-looking
        # `pass@1 = 0.00` that actually means "the harness was broken".
        error=f"{type(error).__name__}: {error}",
    )


def evaluate(
    task_dirs: list[Path], llm: BaseChatModel | None = None, max_steps: int = MAX_STEPS
) -> tuple[AgentResult, ...]:
    """Run the agent over the given tasks, in order, one attempt each.

    Sequential on purpose, and measured rather than assumed: against this Ollama server, three
    requests took 1.7s run one after another and 2.8s run concurrently — a 0.59x "speedup".
    One local model is one set of weights being time-shared, so concurrency here buys nothing
    and costs the per-task timings their meaning. `abatch(max_concurrency=n)` is right there if
    you point this at a hosted model, and wrong for the one in the README.

    Returns the results rather than a report: the suite name belongs to whoever knows which
    suite this is, which is `run_suite`.
    """
    results: list[AgentResult] = []
    for task_dir in task_dirs:
        try:
            results.append(solve_task(task_dir, llm=llm, max_steps=max_steps))
        except Exception as error:  # noqa: BLE001 — one task's crash must not end the suite
            print(f"  {task_dir.name}: run failed — {type(error).__name__}: {error}")
            results.append(crashed(task_dir, error))
    return tuple(results)


def run_suite(suite: str, limit: int = 3, llm: BaseChatModel | None = None) -> int:
    """Run a named suite and write its report. Returns a process exit code.

    Note `limit` defaults to 3 and is applied *after* sorting, so a fourth workshop task is
    silently dropped unless you raise it — pass `--limit 4`.
    """
    if suite == "workshop":
        # Discovered by glob rather than hardcoded, so adding a task directory is enough.
        task_dirs = sorted(p.parent for p in WORKSHOP_TASKS_DIR.glob("*/task.json"))[:limit]
        return _finish(EvalReport("workshop", evaluate(task_dirs, llm=llm)))

    # HumanEvalFix tasks do not exist on disk: they are generated from a vendored JSON subset
    # into a temp directory, used, and thrown away. Keeps a benchmark out of the repo while
    # letting it run through exactly the same task machinery as the workshop fixtures.
    with tempfile.TemporaryDirectory() as temp:
        rows = load_vendored_rows()[:limit]
        task_dirs = [write_task_dir(row, Path(temp)) for row in rows]
        report = EvalReport("humanevalfix", evaluate(task_dirs, llm=llm))

    # Published outside the `with` block: the temp tasks are gone by now, but the results are
    # values, not files, so nothing is lost.
    return _finish(report)


def _finish(report: EvalReport) -> int:
    """Publish the report and decide the process exit code.

    Two behaviours worth having, both learned from watching this go wrong:

    A run where EVERY task crashed has measured nothing, so it must not overwrite the last good
    `results/<suite>.json` — the numbers in that file were real and these are not. It goes to a
    separate `.crashed.json` instead, and the diagnosis goes to stderr where a redirected stdout
    cannot hide it.

    And the exit code is non-zero if anything crashed. `run_suite` used to `return 0`
    unconditionally, so a CI job, a Makefile or an `&&` chain could not tell a broken harness
    from a bad agent — the two things this repo exists to keep apart.
    """
    print(report.format_table())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if report.crashes and len(report.crashes) == len(report.results):
        target = RESULTS_DIR / f"{report.suite}.crashed.json"
        target.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
        print(
            f"\nevery task crashed — nothing was measured, so {report.suite}.json was left "
            f"alone. First reason: {report.crashes[0].error}",
            file=sys.stderr,
        )
        print(f"\nwrote {target}")
        return 1

    _publish(report)
    return 1 if report.crashes else 0


def _publish(report: EvalReport) -> None:
    """Save the JSON, so a run can be compared against a later one.

    Does NOT print the table: `_finish` has already printed it, for both the crashed path and
    this one. It used to print it here as well, so every successful eval reported its results
    twice — harmless, and exactly the kind of thing that survives because nobody reads their own
    output that carefully.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # One file per suite, overwritten each run.
    target = RESULTS_DIR / f"{report.suite}.json"
    target.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
