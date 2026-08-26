# agentgraph (ReAct edition)

A teaching repository for a workshop on how a coding agent actually works. This is the edition
where the agent **reasons before it acts** — built on LangGraph, driven by [JetBrains Mellum2
Thinking](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M) served by
Ollama, fixing real bugs locally for $0.

Three sibling repositories build the same agent, and the interesting content is the step between
them:

| | framework | reasoning |
|---|---|---|
| `agentfix-workshop` | none — a hand-written `for` loop | no |
| `agentfix-langchain` | LangGraph | no |
| **`agentfix-react`** (this one) | LangGraph | **yes** |

The previous edition ended on a measurement and an open question. Its agent solved every task
and reasoned on **0 of 7** turns: seven tool-calling turns carrying no explanation, and the only
prose arriving *after* the fix was already verified. That is the Act-only baseline from the
ReAct paper. This repository closes that gap, and the point of the workshop is that closing it
is one flag plus its consequences — and the consequences are the interesting part.

Every test runs against a scripted fake model, so the repo does not depend on your Ollama setup
working. Real inference is the reward, not a prerequisite.

## What actually changed

One line, in `src/agentgraph/llm/client.py`:

```python
reasoning=True
```

The Thinking model emits `<think>...</think>` whether you ask or not. That flag decides who has
to deal with it: left unset, the tags stay **inline in the answer**, so the next prompt carries
the model's private deliberation, `write_file` receives a "complete file" with a monologue at the
top, and the trace prints it all as if it were the answer. Set, Ollama returns the reasoning on
its own channel and `content` holds only the answer.

Then the consequences, which are not one flag:

- **The trace was lying.** It read reasoning off `content`, which against this model is usually
  empty on a turn that reasoned for three hundred tokens and then called a tool. Ported
  unchanged it reported `(NO REASONING)` on every reasoning turn — confidently backwards. The
  observability did not break loudly when the model changed underneath it; it kept reporting.
  See `agent/trace.py`.
- **A turn with no tool call stopped being rare.** The Instruct model acted on every turn but the
  last. A thinking model will happily spend a whole turn deliberating and ask for nothing, and
  the old answer to a turn like that — nudge it, go again — is an unbounded loop wearing a step
  budget as a disguise. Hence `idle_turns` and `MAX_IDLE_TURNS`: a loop guard for *thinking*,
  beside the one for actions.
- **The action guard had to keep ignoring reasoning.** `call_signature` hashes the tool name and
  arguments only. A small model rarely repeats itself word for word — it reaches the same dead
  end by a slightly different argument each time. Include the reasoning and every repeat looks
  novel, the guard never fires, and the run burns its budget re-reading one file.
- **Reasoning is not free, and it compounds.** It is generated tokens, and LangChain re-sends
  every prior thought on every later turn — so a thought is paid for once when it is generated
  and again on every turn after it. On `01-shopcart` that bought a *shorter* path (6 turns
  against the Act-only agent's 8) for 30% more tokens and 70% more wall clock; on the hardest
  task, one run spent 46k tokens and 8.5 minutes and solved nothing. The numbers, including the
  run-to-run spread, are under [Measured performance](#measured-performance) — and that trade is
  why `agentgraph eval` reports a `thinks` column next to the cost rather than on its own.

## Do I need a special "tool calling" model?

No — and this comes up because the model card looks like it says otherwise. Its serving section
reads:

```bash
# Without tool calling
vllm serve JetBrains/Mellum2-12B-A2.5B-Thinking --max-model-len 131072 --reasoning-parser qwen3
# With tool calling
vllm serve JetBrains/Mellum2-12B-A2.5B-Thinking --max-model-len 131072 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

That is **one set of weights**. Tool calling is native to the model — the card benchmarks it on
BFCL v3/v4 — and those flags only tell *vLLM* how to **parse** what comes back: `qwen3` for the
`<think>` blocks, `hermes` for the tool-call syntax. Omit them and vLLM hands you raw text with
the tags still in it.

We use neither, because we do not use vLLM. Ollama's chat template does that parsing
server-side, and `ChatOllama` hands the results back on separate channels. Verified on this
machine, one turn, one tool bound:

```
tool_calls        [{'name': 'run_tests', 'args': {}, ...}]
content           ''
reasoning_content "Okay, the user wants me to fix a failing test. But wait, I need to figure
                   out which test is failing..."
```

Which is the same lesson as `ChatOllama`-over-`ChatOpenAI` one layer up: pick the right
integration and the parsing is free. `agentgraph doctor` checks both channels for you.

## Which setup option should you use?

| Option | Who | RAM | Model |
|---|---|---|---|
| 1 (default) | 16 GB+ laptop | 16 GB+ | Mellum2 12B **Thinking** via Ollama (~8 GB download) |
| 2 | weaker laptop | ~4 GB | `qwen3:1.7b` (~1.4 GB) |
| 3 | browser only | any | Google Colab — `notebooks/agentgraph.ipynb` |

**Option 2 must be a reasoning model.** The previous edition's fallback, `qwen2.5-coder:1.5b`,
has no thinking mode: point this repo at it and every run still completes, silently, as the
previous workshop's Act-only agent — and "this agent does not reason" becomes a fact about your
setup rather than about the model. `qwen3:1.7b` is the smallest thing that both thinks and calls
tools. `doctor` fails rather than letting this pass quietly.

Options 1 and 2 run on macOS, Linux, WSL2 and native Windows. Option 3 needs only a browser.

**Windows users: prefer WSL2.** The sandbox that executes the agent's test runs is POSIX-shaped.

All measurements in this README were taken on macOS with Option 1 unless stated otherwise. Where
a path is untested, it says so.

## Setup

### Step 1 — install `uv` and Ollama

<details open>
<summary><b>macOS</b> (verified)</summary>

```bash
brew install uv ollama
ollama serve &                  # or: open -a Ollama   (the app starts the same server)
```

Homebrew's `ollama` and Ollama.app are the same server on `localhost:11434` — use either, not
both. Without Homebrew: `curl -LsSf https://astral.sh/uv/install.sh | sh` and Ollama from
[ollama.com/download](https://ollama.com/download).
</details>

<details>
<summary><b>Linux</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh
```

The install script registers a systemd service, so the server is already listening;
`systemctl status ollama` confirms it. If you installed the tarball by hand, run `ollama serve`
in its own terminal. A GPU is not required — CPU inference works, just slower. Reasoning makes
that difference more noticeable than it was in the previous edition: a thinking turn generates
several hundred tokens before it acts.
</details>

<details>
<summary><b>Windows — WSL2 (recommended)</b></summary>

In PowerShell, once:

```powershell
wsl --install -d Ubuntu
```

Then follow the **Linux** instructions inside the Ubuntu shell and do everything — `git clone`,
`uv`, `ollama`, the runs — inside WSL2. Keep the clone on the Linux filesystem
(`~/agentfix-react`, not `/mnt/c/...`); test runs across the `/mnt/c` bridge are slow enough to
be annoying.

WSL2 takes a fraction of your RAM by default (50%, capped at 8 GB on older builds), and that
fraction — not your machine's spec sheet — has to hold an 8 GB model. If `free -g` inside WSL2
shows under 16 GB, raise it in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=16GB
```

then `wsl --shutdown` in PowerShell and reopen the shell.
</details>

<details>
<summary><b>Windows — native PowerShell</b> (sandbox untested)</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install --id Ollama.Ollama      # or the installer from ollama.com/download
```

The installer runs Ollama in the background, so the server is already on `localhost:11434` (look
for the tray icon). Every `uv run ...` command below is identical in PowerShell, and forward
slashes in task paths are fine.

Two caveats: `agentgraph doctor` cannot read RAM on Windows and skips that check rather than
failing it, and the subprocess sandbox has not been run on native Windows. If `doctor` reports a
`sandbox` failure, switch to WSL2 rather than debugging it during the workshop.
</details>

### Step 2 — get the model

<details open>
<summary><b>Option 1 — Mellum2 Thinking (16 GB+ RAM)</b></summary>

```bash
ollama pull hf.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M
ollama create agentgraph-mellum2-thinking -f Modelfile
```

Note **Thinking**, not the Instruct model the previous workshop used. The `create` step derives a
model with `num_ctx 16384` baked in and gives it the short name `DEFAULT_MODEL` in
`src/agentgraph/config.py` expects. There is nothing extra to pull or enable for tool calling —
see [above](#do-i-need-a-special-tool-calling-model).
</details>

<details>
<summary><b>Option 2 — the 1.4 GB fallback</b></summary>

```bash
ollama pull qwen3:1.7b
export MELLUM_MODEL=qwen3:1.7b     # PowerShell: $env:MELLUM_MODEL="qwen3:1.7b"
```

No `ollama create` step: the client sends `num_ctx` with every request and Ollama's native API
honours it. Set `MELLUM_MODEL` in every shell you use, or put it in your shell profile.

A 1.7B model fixes fewer bugs than Mellum2 and gets its tool-call JSON wrong more often. That is
not a broken setup — it is why the loop guards exist. Untested by the author for solve rates;
the reasoning and tool-calling channels were verified.
</details>

<details>
<summary><b>Option 3 — Google Colab</b></summary>

Open `notebooks/agentgraph.ipynb` in Colab and run the cells in order. It installs Ollama, pulls
`qwen3:1.7b`, clones this repo, disables pushing, and runs the agent from a cell.
</details>

### Step 3 — install and check

```bash
uv sync --extra dev
uv run agentgraph doctor
```

`doctor` is the fastest way to find a broken setup, because almost every failure here produces a
symptom that looks like something else — a too-small context window looks like a stupid model,
not a misconfiguration.

Two of its checks are new in this edition, and they are the ones worth having, because both
failures leave you with a *working* agent that nothing else will complain about:

- **`reasoning`** — the model thinks, and the thinking arrives on its own channel. Fails
  distinctly if the reasoning is coming back inline as `<think>` tags, which means
  `reasoning=True` is not reaching the server.
- **`tool calling`** — it can still act while thinking. A model that reasons and calls nothing
  changes no files.

A healthy Option 1 machine reports:

```
[PASS] python: 3.13.14
[PASS] ram: 24.0 GB total, 9.8 GB free
[PASS] ollama installed: /usr/local/bin/ollama
[PASS] ollama server: reachable at http://localhost:11434
[PASS] model present: agentgraph-mellum2-thinking
[PASS] generation: 41 tok/s (874 tokens in 21.4s)
[PASS] context window: 16384 tokens
[PASS] reasoning: 628 chars of thinking returned
[PASS] tool calling: requested calculate
[PASS] sandbox: executes tests

READY 41 tok/s (874 tokens in 21.4s)
```

## Use

```bash
uv run agentgraph solve tasks/workshop/01-shopcart --verbose
uv run agentgraph eval --suite workshop
uv run agentgraph eval --suite humanevalfix --limit 5
```

`--verbose` prints the trace live: one line per model turn, one per tool call, and an indented
`thinks` line carrying the reasoning behind each decision.

## Tests

unittest only, no pytest anywhere — including inside the task fixtures the agent fixes.

```bash
uv run python -m unittest discover -s tests -t .          # 221 tests, offline, ~5s
uv run python -m unittest tests.test_reasoning -v         # just the ReAct behaviour
AGENTGRAPH_LLM_TESTS=1 uv run python -m unittest discover -s tests -t .   # + live-model tests
```

The whole suite runs with no model process anywhere: `llm/fake.py` is a real `BaseChatModel`
returning a scripted list of replies, so the tests drive the **real** graph against the **real**
tools in a real temp directory. Only the model is replaced.

Crucially the fake puts reasoning in *exactly* the field the real client uses
(`additional_kwargs["reasoning_content"]`). A fake that put it anywhere else would let a broken
agent pass — the thinking guard, both nudges and the trace's `thinks` line would all be tested
against a field the real model never populates.

`uv sync --extra dev --extra prebuilt` additionally enables `tests/test_prebuilt.py`.

## Reading order

1. `tools/base.py` — what a tool is, the limits on what it may return, and the artifact channel
2. `tasks/loader.py` — what a task is; the copy-to-tempdir context manager
3. `tools/fs.py` — `list_files`, `read_file`, `write_file`
4. `tools/tests_tool.py` — `run_tests`, the agent's only oracle
5. `llm/client.py` — **the one flag, and what it costs**
6. `agent/state.py` — what the graph carries between nodes, and the reducers that combine it
7. `agent/graph.py` — **the agent.** If you read one file, read this one.
8. `runner.py` — how the pieces are wired together

Then `agent/trace.py` (observability — and the one file the new model actively broke), `llm/fake.py`,
`sandbox/`, `eval/`, and `doctor.py`.

`agent/prebuilt.py` is the argument rather than the implementation: the same agent built from
`langchain.agents.create_agent` and its middleware, with each claim carrying the measurement
behind it. Needs `--extra prebuilt`.

## What the framework gave us

- `ToolNode` replaces the hand-written `dispatch`, including its unknown-tool and bad-argument
  observations, and answering several calls in one turn.
- `add_messages` makes the history append-only by construction.
- Reducers on the rest of the state — `operator.add` for the counters, a two-argument
  `keep_larger` for the peak — so a node returns a delta and never reads the old value.
- Callbacks carry the trace, reasoning included. `agent/trace.py` is a `BaseCallbackHandler`
  handed to the graph once, so the nodes contain no tracing code at all.
- Checkpointing: `InMemorySaver` snapshots the state after every node.
- `ChatOllama` parses tool calls, token usage, malformed arguments — **and separates the
  reasoning from the answer.** Nothing in this repo parses a `<think>` tag.
- Reasoning is a property of the client, not the loop, so `agent/prebuilt.py` inherits a
  reasoning model for free. That is the framework getting something right, and worth saying.

## What it did not

- **`handle_tool_errors` defaults to letting a tool's exception kill the run.** You have to opt
  back in — and passing a *string* rather than `True` silently discards the specific error, so
  the model stops being told which argument it forgot.
- **Neither shape of bad tool-call JSON is handled for you.** `invalid_tool_calls` gets no reply
  message at all, though the API requires an answer to every call — and a reply the client cannot
  parse raises straight through the graph and ends the run. Both are ours to catch, in
  `tools_node` and `agent_node` respectively.
- **Neither loop guard.** LangGraph has no hook for either. LangChain 1.x gives you a seam for
  the action guard (`wrap_tool_call`) but not the policy — and for the *thinking* guard it gives
  you no good seam at all: `after_model` could count idle turns, but the counter would live on
  the middleware instance, so it would not survive a checkpoint and would leak into the next
  run. `AgentState.idle_turns` is scoped to the run because the state is.
- **The step budget — on LangGraph.** `recursion_limit` counts node executions, not model turns.
  On LangChain 1.x this one has *moved*: `ModelCallLimitMiddleware(run_limit=N)` counts exactly
  what `AgentState.step` counts. See `agent/prebuilt.py`, including the measurement showing it
  is silently ignored if you order the middleware wrong.
- **Checkpointing is only as good as what you put in the state.** The test verdict used to live
  on the `run_tests` tool object. The graph was resumable; the agent was not — a resumed run
  rebuilt that tool empty and reported a solved task unsolved. The verdict now travels as a
  `ToolMessage` artifact into `AgentState.tests_passed`.
- **Tool calls in one turn run concurrently by default.** `ToolNode` batches through a real
  `ThreadPoolExecutor` even when nothing asked for it, so a `run_tests` in the same message as a
  `write_file` can measure the file as it was *before* the write. Message order is preserved, so
  the trace looks innocent. `max_concurrency=1` restores one-at-a-time execution.
- **The wrong integration will lie to you.** An earlier version used `ChatOpenAI` against
  Ollama's `/v1` endpoint to keep the wire format byte-identical to the no-framework original.
  Two of the three settings that decide whether the agent works were being discarded in transit,
  silently. Measured, same server:

  | | `ChatOpenAI` via `/v1` | `ChatOllama` |
  |---|---|---|
  | cap on one reply | `max_completion_tokens=8` → 692 tokens | `num_predict=8` → 8 tokens |
  | context window | `options` dropped; `ollama ps` says 4096 | `num_ctx=8192` → `ollama ps` says 8192 |

  A compatibility endpoint accepts the requests it does not honour — and it has no concept of
  `think` at all, so this edition could not have been built on it.

## The context window

The single most consequential setting, and the one nothing else will tell you about. Too small a
window does not error — it silently truncates the middle of the agent's history, which looks like
a stupid model rather than a misconfigured one. `agentgraph doctor` checks it against
`MIN_CONTEXT_LENGTH` and fails if the loaded model reports less.

Reasoning raises the stakes here. Prior thoughts are re-sent on every later turn, so context
grows faster than it did in the Instruct edition: measured peak across the workshop suite went
from **1,574** tokens to **6,163**, and the worst single run observed reached **9,003** — over
half the window, on a three-file project. 16384 still holds these tasks, but the headroom is a
fraction of what it was, and a harder task is where the truncation would begin.

`max_tokens` also had to rise from 1024 to 4096, because one reply is now the reasoning *plus* a
complete file. A reply truncated mid-thought loses the tool call at the end of it, which presents
as a model that inexplicably stopped acting.

## Measured performance

Option 1, macOS, 24 GB RAM, 41 tok/s. **Two consecutive runs of the same suite, unchanged:**

```
run 1
task                     solved   steps   thinks   tokens    peak ctx   seconds
--------------------------------------------------------------------------------
01-shopcart              True     6       6/6      11151     2612       45.12
02-invoice               True     7       7/7      14311     2902       59.17
03-parser                False    9       9/9      46203     9003       510.41
pass@1 = 0.67  (3 task(s))  peak prompt = 9003 tok  reasoning on 22/22 turns

run 2
task                     solved   steps   thinks   tokens    peak ctx   seconds
--------------------------------------------------------------------------------
01-shopcart              True     9       9/9      34061     6163       134.78
02-invoice               True     7       7/7      15103     3216       95.41
03-parser                True     9       9/9      21793     2506       207.61
pass@1 = 1.00  (3 task(s))  peak prompt = 6163 tok  reasoning on 25/25 turns
```

Both runs are reported because **one of them would have been a lie.** Same code, same model,
same tasks: `pass@1` was 0.67 and then 1.00, and `03-parser` went from failing after 510 seconds
and 46k tokens to passing in 208. Run it a third time in isolation and it solved in 20 seconds
and 10k tokens.

Reasoning on **every** turn in both runs, which is the headline — the previous edition managed 0
of 7. But the rest of the comparison is not a win, and pretending otherwise would waste the
measurement. Eval-to-eval on `01-shopcart`, against the Instruct edition:

| | Instruct (previous edition) | Thinking, run 1 | Thinking, run 2 |
|---|---|---|---|
| verdict | SOLVED | SOLVED | SOLVED |
| steps | 8 | **6** | 9 |
| turns with reasoning | 0 of 7 | 6 of 6 | 9 of 9 |
| tokens | 8,566 | 11,151 | 34,061 |
| peak context | 1,387 | 2,612 | 6,163 |
| seconds | 26.1 | 45.1 | 134.8 |

Read the `steps` row and the `tokens` row together, because they disagree. Run 1 reached the fix
in **six** turns where the Act-only agent needed eight — reasoning genuinely bought a shorter
path, which is the ReAct claim working. It still cost 30% more tokens and 70% more wall clock to
get there, because the turns it saved were cheap and the turns it added were not.

And run 2 is the same agent taking nine turns and three times the tokens for the same fix. So:

- **Reasoning is not free, and its cost has a long tail.** The worst observed task spent 46k
  tokens and 8.5 minutes and produced nothing. Nothing bounds how long a chain of thought
  becomes — only `max_tokens` per reply and the step budget, both blunt.
- **Variance went up, not just cost.** `temperature=0.6` is JetBrains' published setting for this
  checkpoint and the previous edition used it too, but a long chain of thought amplifies one
  unlucky token into a whole wrong plan, so the spread is wider here than it was. If you need
  numbers you can compare, set `temperature=0.0` in `LLMConfig` — and accept that a stuck model
  then has no way out of repeating itself.
- **`pass@1` from a 3-task suite and one attempt each is a noisy statistic.** It was noisy in the
  previous edition too; reasoning just made it obvious. Two runs is not a measurement either —
  it is enough to know that one run is not.

Eval is deliberately sequential, and that is measured rather than assumed: against this Ollama
server, three requests took 1.7s run one after another and 2.8s run concurrently. One local
model is one set of weights being time-shared.

## Running things in Docker

The default sandbox is a hardened subprocess: stripped environment, resource limits, a timeout.
It is **not** a security boundary — test code runs as your user, on your machine. For real
isolation:

```bash
docker build -t agentgraph-sandbox -f Dockerfile.sandbox .
AGENTGRAPH_SANDBOX=docker uv run agentgraph solve tasks/workshop/01-shopcart --verbose
```

PowerShell wants `$env:AGENTGRAPH_SANDBOX="docker"` on its own line first. The container mounts the
workspace read-only, runs as a non-root user, and has no network. Note that `Dockerfile.sandbox`
installs nothing — `unittest` is in the standard library, so there is no version to pin and no
drift between the host and the container to catch.

Docker execution is untested by the author on this edition; the backend's own tests
(`tests/test_docker_backend.py`) assert the command line rather than starting containers, which
is what keeps them runnable everywhere.

## Platform notes

- **RAM check**: `doctor` reads available memory on macOS and Linux only. On Windows it skips the
  check rather than failing it.
- **The sandbox**: `subprocess_backend.py` uses POSIX resource limits. Untested on native Windows.
- **Case-insensitive filesystems**: macOS lets `Tests/TEST_CART.PY` address the same file as
  `tests/test_cart.py`, so the check protecting the agent's own oracle from the agent is
  deliberately case-insensitive. There is a reproduced-escape test for it.

## Known limitations

- One attempt per task, no retries and no best-of-n. `pass@1` means exactly that.
- The agent rewrites whole files rather than emitting diffs. At this model size a diff-based tool
  contract is one the model cannot satisfy, which looks exactly like a broken agent.
- Nothing stops the agent from writing code that special-cases the test inputs. The write
  allow-list and the protected test suite close the routes that were actually reproduced; that
  one stays open.
- **Reasoning tokens cannot be counted separately.** Ollama reports one `output_tokens` covering
  the thinking and the answer together, so `reasoning_turns` counts *turns*, not tokens. A
  per-turn reasoning cost would be a number we made up.
- **Nothing checks whether the reasoning is any good.** A model can reason fluently to the wrong
  conclusion, and this agent will follow it there. The tests are the only thing that catches
  that — which is the same guarantee as the previous edition, doing more work than before.
- **Reasoning leaking inline is caught once, at setup, not per turn.** `agentgraph doctor` detects
  reasoning arriving as `<think>` tags in the answer instead of on its own channel. Nothing
  re-checks it mid-run, so if a server stopped honouring `think` partway through, the trace would
  fold the monologue into the action summary and `reasoning_turns` would undercount. Deliberately
  not fixed: the fix is a `<think>` parser, and the claim that nothing in this repo parses one is
  worth more than defending against a misconfiguration `doctor` already names.
- **A reply the client cannot parse costs a turn, not the run — but it does cost a turn.**
  `ChatOllama` never reports an invalid tool call: measured, it either keeps bad arguments
  leniently or raises `OutputParserException`. That used to propagate and have the task recorded
  as a CRASH; `agent_node` now catches it, tells the model what was wrong and asks again, bounded
  by the same guard as thinking. It is still a wasted turn out of the budget, and a model that
  cannot emit valid JSON twice running is still abandoned. Consequently the `invalid_tool_calls`
  handling in `agent/graph.py` is defensive code for a backend this repo is not currently
  using — its docstring says so.

## The workshop exercises

Not in this repository yet. The two functions the previous edition had students write —
`route_after_agent` and the loop guard — are the two this edition had to change, which is what
makes them the natural exercises here. They will land as `exercises/`, matching the sibling
repos' layout.
