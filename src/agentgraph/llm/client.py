"""The real model client: a `ChatOllama` pointed at the local Ollama server.

The only file in the project that performs network I/O, and still one function — because the
framework's job is exactly this. The client speaks Ollama's protocol, parses tool calls into
`AIMessage.tool_calls`, records token usage in `usage_metadata`, and — new in this edition —
separates the model's reasoning from its answer. The no-framework edition hand-wrote all of
that. What it does NOT do is rescue unparseable tool arguments: those raise, and
`agent/graph.py` catches it.

## The one line that makes this the ReAct edition

    reasoning=True

The model is the Thinking checkpoint, so it emits `<think>...</think>` before it answers
whether we ask for it or not. `reasoning` decides who has to deal with those tags:

    reasoning=None (default)  the tags stay INLINE in `AIMessage.content`. Your prompt now
                              contains the model's private deliberation, `write_file` gets a
                              "complete file" with a monologue at the top of it, and the trace
                              prints the whole thing as if it were an answer.
    reasoning=True            Ollama is asked to `think`, and the reasoning comes back on its
                              own channel: `AIMessage.additional_kwargs["reasoning_content"]`,
                              with `content` holding only the answer.

So the flag is not "switch thinking on" — the Thinking model thinks either way. It is "give me
the thinking as structured data instead of as text I have to strip". That is the same lesson as
the `ChatOpenAI`-vs-`ChatOllama` story below, one layer up: the right integration parses the
model's output so you do not have to.

Measured with qwen3:1.7b, one turn, tools bound:

    tool_calls        [{'name': 'run_tests', 'args': {}, ...}]
    content           ''
    reasoning_content "Okay, the user wants me to fix the failing test. Let me see what tools
                       I have available..."

Note that `content` is EMPTY while the model is plainly reasoning. This is why
`agent/trace.py` had to change: the previous edition read reasoning off `content`, so against
this model it would report `(NO REASONING)` on every acting turn — the exact opposite of what
happened.

## The cost, which is not free

LangChain sends prior reasoning back: converting the history for the next request copies each
`reasoning_content` into a `thinking` field on that turn. So every thought the model has ever
had is re-sent on every subsequent turn, and thoughts are the largest part of a reply. Context
grows faster here than it did in the Instruct edition, which is why `max_tokens` had to grow
and why the peak-context column in `agentgraph eval` is worth watching.

"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from agentgraph.config import LLMConfig


def make_chat_model(config: LLMConfig | None = None) -> ChatOllama:
    """Build the model client. No api_key: Ollama has no authentication to satisfy."""
    config = config or LLMConfig.from_env()
    return ChatOllama(
        base_url=config.api_url,
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        # Ollama's names for the two caps. `num_predict` is the ceiling on ONE reply — now the
        # reasoning and the answer together — and `num_ctx` is the context window the model is
        # loaded with. Both honoured here, which is the entire reason this file uses ChatOllama.
        num_predict=config.max_tokens,
        num_ctx=config.num_ctx,
        # Reaches the server as Ollama's `think`. See the module docstring: this is what puts
        # the reasoning on its own channel instead of inline in `content`.
        reasoning=config.reasoning,
    )
