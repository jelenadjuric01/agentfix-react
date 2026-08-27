"""Settings: where the model is, which model, and how to sample from it."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


# Task fixtures and result files live with the code, not in whatever directory the student
# happened to be standing in when they ran the CLI — so this is derived from `__file__` rather
# than from the working directory.
#
# Found by climbing rather than by counting levels, because this package ships in two shapes and
# a fixed `.parents[n]` cannot be right for both: `src/agentgraph/` here, and `agentgraph/` at
# the root of a JetBrains Academy task directory, which is one level shallower. Counting levels
# in the Academy layout resolved to the LESSON directory, where `tasks/` does not exist — so
# `eval` reported no fixtures and wrote its results nowhere near the repo.
def _find_root() -> Path:
    """The nearest directory ABOVE this package that holds the task fixtures.

    `parents[1:]` skips the package directory itself, which has a `tasks/` of its own —
    `tasks/loader.py`, the module that says what a task is. Searching from the package would
    match that subpackage and never look further.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents[1:]:
        if (candidate / "tasks").is_dir():
            return candidate
    # No fixtures anywhere above us: an installed wheel, or a checkout with `tasks/` removed.
    # The src-layout answer is the best guess left, and every caller is about to report a
    # missing path with it in the message.
    return here.parents[2] if len(here.parents) > 2 else here.parent


REPO_ROOT = _find_root()

# Ollama's own API root — not the `/v1` compatibility endpoint. `ChatOllama` speaks the
# native protocol, which is the only one that honours `num_ctx`, `num_predict` and `think`;
# see llm/client.py for what the `/v1` endpoint silently discarded.
DEFAULT_BASE_URL = "http://localhost:11434"

# The THINKING checkpoint, which is the whole point of this edition. Same 12B/A2.5B weights as
# the Instruct model the previous workshop used, trained to emit its reasoning inside
# `<think>...</think>` before it answers. Tool calling is native to it — there is no separate
# "with tools" build to pull. (The `--tool-call-parser` / `--reasoning-parser` flags on the
# model card are vLLM's instructions for how to PARSE that output; Ollama's chat template does
# that job for us, which is why they do not appear anywhere in this repo.)
BASE_MODEL = "hf.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M"
DEFAULT_MODEL = "agentgraph-mellum2-thinking"

# The tier-2 model, for a laptop that cannot hold 8 GB of weights. It has to REASON, which
# rules out the `qwen2.5-coder:1.5b` the previous edition fell back to: a model with no
# thinking mode turns this agent back into the Act-only one from the last workshop, and every
# reasoning-shaped thing in the trace silently disappears. Qwen3 is the smallest thing that
# both thinks and calls tools.
FALLBACK_MODEL = "qwen3:1.7b"

# `agentgraph doctor` fails if the loaded model reports less than this. A too-small context
# does not error — it silently truncates the middle of the agent's history, which looks like
# a stupid model rather than a misconfiguration.
MIN_CONTEXT_LENGTH = 16384


@dataclass(frozen=True)
class LLMConfig:
    """How to reach the model, and how to sample from it. Frozen: read-only after creation."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    # JetBrains' published settings for the Thinking checkpoint, not a sweep on this project.
    # `top_k` is new here — the Instruct edition left it alone.
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    # Cap on ONE reply, and it had to grow. In the Instruct edition 1024 tokens was a ceiling
    # on one thing: the largest complete file `write_file` could emit. Now one reply is the
    # reasoning PLUS that file, and reasoning on a small model is not short — measured on
    # 01-shopcart, a thinking turn spends a few hundred tokens before it acts. Too low a cap
    # here truncates the reply, and a reply cut off mid-thought loses the tool call at the end
    # of it: the model appears to stop acting for no reason. Reaches the server as Ollama's
    # `num_predict`.
    max_tokens: int = 4096

    # The context window the model is loaded with. Honoured because the client speaks Ollama's
    # native API, which is why `agentgraph doctor` can check it and expect to be obeyed.
    num_ctx: int = 16384

    # Ask the model to think, and to hand the thinking back on its own channel rather than
    # inline in the answer. See llm/client.py — this one flag is the difference between this
    # edition and the last one.
    reasoning: bool = True

    @property
    def api_url(self) -> str:
        """`base_url` with a trailing `/v1` removed, if someone's environment still has one.

        Tolerance rather than a second endpoint: every request this project makes goes to
        Ollama's native API, so a `MELLUM_BASE_URL` left over from the `/v1` days would
        otherwise produce URLs like `.../v1/api/ps`.
        """
        trimmed = self.base_url.rstrip("/")
        return trimmed[: -len("/v1")].rstrip("/") if trimmed.endswith("/v1") else trimmed

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Defaults, overridden by MELLUM_BASE_URL and MELLUM_MODEL if they are set."""
        return replace(
            cls(),
            base_url=os.environ.get("MELLUM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("MELLUM_MODEL", DEFAULT_MODEL),
        )
