"""Preflight checks: `agentgraph doctor`.

A workshop-specific tool, and the reason is worth understanding. Almost every way this setup
fails produces a symptom that looks like something else:

  - too small a context window      -> history is silently truncated; looks like a dumb model
  - the base model, not the derived -> same thing, because num_ctx lives in the Modelfile
  - Ollama not running              -> a connection error deep inside the SDK
  - not enough free RAM             -> the model loads, then everything is extremely slow
  - a model that does not think     -> the agent still runs, and is silently the PREVIOUS
                                       workshop's Act-only agent
  - a model that cannot call tools  -> it reasons beautifully and changes no files

Each check below turns one of those into a named diagnosis with the command that fixes it.
Fifteen minutes of a two-hour workshop can disappear into any one of them.

The last two are new in this edition and are the ones worth having. Both failures leave you
with a *working* agent, which is why nothing else will report them: point this project at an
Instruct model and every run still completes, just with no reasoning anywhere — and the honest
conclusion "this agent does not reason" would be a fact about your setup, not about the model.
Both are checked with one real model call, because only a real call can answer either.

Every check returns a `Check` rather than raising, so one failure never hides the others — you
get the full picture in one run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentgraph.config import BASE_MODEL, FALLBACK_MODEL, MIN_CONTEXT_LENGTH, LLMConfig

PULL_HINT = f"run: ollama pull {BASE_MODEL}"
CREATE_HINT = "run: ollama create agentgraph-mellum2-thinking -f Modelfile"
SERVE_HINT = (
    "the Ollama server is not answering — start it "
    "(macOS: open the Ollama app; otherwise: ollama serve)"
)

MIN_TOTAL_RAM_BYTES = 16 * 1024**3
COMFORTABLE_FREE_RAM_BYTES = 9 * 1024**3

# One short call measures cold model load and prefill, not generation. So warm up first, throw
# that call away, then time a prompt long enough that per-call overhead stops dominating.
WARMUP_PROMPT = "Reply with: ok"
THROUGHPUT_PROMPT = "Count from 1 to 120, one number per line, digits only."


@dataclass(frozen=True)
class Check:
    """One check's verdict. `detail` carries the fix, not just the symptom."""

    name: str
    ok: bool
    detail: str


def _get_json(url: str, timeout_s: float = 5.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    # json.loads returns Any; every caller indexes this like an object, so a list or a bare
    # string from an unexpected endpoint should read as "no data", not crash later.
    return payload if isinstance(payload, dict) else None


def _check_python() -> Check:
    version = sys.version_info
    ok = (version.major, version.minor) >= (3, 12)
    return Check("python", ok, f"{sys.version.split()[0]}" + ("" if ok else " — need >= 3.12"))


def _capture(command: list[str]) -> str:
    """stdout of a successful command. Raises CalledProcessError/OSError otherwise."""
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def _memory_bytes() -> tuple[int | None, int | None]:
    """(total, free). Either may be None on a platform we cannot read.

    Hand-rolled per platform rather than pulling in psutil, to keep the dependency list short.
    Returning None instead of raising matters: a machine whose memory we cannot read is not a
    machine that fails the check.
    """
    if sys.platform.startswith("linux"):
        try:
            fields = dict(
                (line.split(":", 1)[0], int(line.split()[1]) * 1024)
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
                if ":" in line and line.split()[1].isdigit()
            )
        except OSError:
            return None, None
        return fields.get("MemTotal"), fields.get("MemAvailable")

    if sys.platform == "darwin":
        try:
            total = int(_capture(["sysctl", "-n", "hw.memsize"]))
            page = int(_capture(["sysctl", "-n", "hw.pagesize"]))
            counts = {
                line.split(":")[0].strip(): int(line.split(":")[1].strip().rstrip("."))
                for line in _capture(["vm_stat"]).splitlines()
                if ":" in line and line.split(":")[1].strip().rstrip(".").isdigit()
            }
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None, None
        reclaimable = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        return total, sum(counts.get(key, 0) for key in reclaimable) * page

    return None, None


def _check_ram() -> Check:
    total, free = _memory_bytes()
    if total is None:
        return Check("ram", True, "could not read memory on this platform — check by hand")

    detail = f"{total / 1024**3:.1f} GB total"
    if free is not None:
        detail += f", {free / 1024**3:.1f} GB free"

    if total < MIN_TOTAL_RAM_BYTES:
        return Check(
            "ram",
            False,
            f"{detail} — under 16 GB, use {FALLBACK_MODEL} or the Colab notebook; see README.md",
        )
    if free is not None and free < COMFORTABLE_FREE_RAM_BYTES:
        detail += " — tight for an 8 GB model unless it is already loaded; close other apps"
    return Check("ram", True, detail)


def _check_ollama_installed() -> Check:
    path = shutil.which("ollama")
    return Check(
        "ollama installed", path is not None, path or "not found — install from ollama.com"
    )


def _check_server(config: LLMConfig) -> Check:
    payload = _get_json(f"{config.api_url}/api/tags")
    if payload is None:
        return Check("ollama server", False, SERVE_HINT)
    return Check("ollama server", True, f"reachable at {config.api_url}")


def _check_model_present(config: LLMConfig) -> Check:
    payload = _get_json(f"{config.api_url}/api/tags") or {}
    names = [model.get("name", "") for model in payload.get("models", [])]
    if any(name.split(":")[0] == config.model for name in names):
        return Check("model present", True, config.model)

    if any(name.split(":")[0] == BASE_MODEL for name in names):
        return Check(
            "model present", False, f"{BASE_MODEL} is pulled but not derived — {CREATE_HINT}"
        )
    return Check(
        "model present", False, f"{config.model} missing — {PULL_HINT}, then {CREATE_HINT}"
    )


def _check_generation(config: LLMConfig) -> Check:
    from agentgraph.llm.client import make_chat_model

    client = make_chat_model(config)
    try:
        client.invoke(WARMUP_PROMPT)
    except Exception as error:
        return Check("generation", False, f"{type(error).__name__}: {error}")

    try:
        started = time.time()
        reply = client.invoke(THROUGHPUT_PROMPT)
        elapsed = max(time.time() - started, 1e-6)
    except Exception as error:
        return Check("generation", False, f"{type(error).__name__}: {error}")

    usage: dict[str, Any] = dict(reply.usage_metadata or {})
    completion = int(usage.get("output_tokens", 0))
    rate = completion / elapsed
    return Check("generation", True, f"{rate:.0f} tok/s ({completion} tokens in {elapsed:.1f}s)")


# A question with an unmistakable answer, so a model that reasons cannot answer it without
# thinking and a model that does not reason still answers it. The tool exists to be called: the
# point is not the arithmetic, it is which channels come back populated.
REASONING_PROMPT = "What is 17 * 23? Use the calculate tool."


def _check_reasoning_and_tools(config: LLMConfig) -> list[Check]:
    """One model call, two verdicts: does it think, and can it still act while thinking?

    Both are checked together because both are answers to the same request, and a local model
    is slow enough that asking twice is a real cost. Returning a list rather than one merged
    Check keeps the diagnosis specific — "it reasons but never calls tools" and "it calls tools
    but never reasons" need completely different fixes, and a single combined FAIL line would
    tell you neither.
    """
    from langchain_core.tools import tool

    from agentgraph.llm.client import make_chat_model

    @tool
    def calculate(expression: str) -> str:  # noqa: ARG001 — never executed, only offered
        """Evaluate an arithmetic expression."""
        # The body is unreachable in this check: we look at whether the model ASKED for the
        # tool, not at what it would have returned. A schema is all the model ever sees.
        return "391"

    try:
        reply = make_chat_model(config).bind_tools([calculate]).invoke(REASONING_PROMPT)
    except Exception as error:
        failed = f"{type(error).__name__}: {error}"
        return [Check("reasoning", False, failed), Check("tool calling", False, failed)]

    # Read from additional_kwargs, exactly where the agent reads it — see agent/trace.py. If
    # this is empty but the answer text contains "<think>", the model IS thinking and the
    # reasoning is arriving inline instead of on its own channel, which means `reasoning=True`
    # is not reaching the server. That is a different bug and worth naming separately.
    thinking = str(reply.additional_kwargs.get("reasoning_content") or "").strip()
    text = reply.text or ""
    if thinking:
        reasoning_check = Check("reasoning", True, f"{len(thinking)} chars of thinking returned")
    elif "<think>" in text:
        reasoning_check = Check(
            "reasoning",
            False,
            "the model is thinking but the reasoning arrived INLINE in the answer — "
            "`reasoning=True` is not reaching the server; see src/agentgraph/llm/client.py",
        )
    else:
        reasoning_check = Check(
            "reasoning",
            False,
            f"{config.model} returned no reasoning — this is probably an Instruct model, not a "
            f"Thinking one. Without it this agent is the previous workshop's. {CREATE_HINT}",
        )

    calls = reply.tool_calls
    tools_check = Check(
        "tool calling",
        bool(calls),
        (
            f"requested {calls[0]['name']}"
            if calls
            else f"{config.model} answered without calling the tool it was given — it cannot act, so "
            "every run will reason and then change nothing"
        ),
    )
    return [reasoning_check, tools_check]


def _check_context(config: LLMConfig) -> Check:
    """The single most consequential setting, and the one nothing else will tell you about."""
    payload = _get_json(f"{config.api_url}/api/ps")
    if payload is None:
        return Check("context window", False, f"cannot read /api/ps — {SERVE_HINT}")

    loaded = [
        model
        for model in payload.get("models", [])
        if model.get("name", "").split(":")[0] == config.model
    ]
    if not loaded:
        return Check(
            "context window",
            False,
            f"{config.model} is not loaded, so its context cannot be read — "
            "run this check again after a generation succeeds",
        )

    length = loaded[0].get("context_length") or 0
    if length < MIN_CONTEXT_LENGTH:
        return Check(
            "context window",
            False,
            f"{length} tokens — too small; the agent's history will be silently truncated "
            f"mid-run. {CREATE_HINT}",
        )
    return Check("context window", True, f"{length} tokens")


def _check_sandbox() -> Check:
    """Prove the execution backend works by actually running a trivial passing test.

    Checked because `run_tests` is the agent's only oracle: if execution is broken, every run
    reports NOT SOLVED no matter how good the model is. Uses whichever backend is configured,
    so it also catches "AGENTGRAPH_SANDBOX=docker but the image was never built".
    """
    import tempfile

    from agentgraph.sandbox.base import get_backend

    with tempfile.TemporaryDirectory() as temp:
        # At the top level, so `discover` needs no package — see write_task_dir.
        Path(temp, "test_ok.py").write_text(
            "import unittest\n\n\nclass T(unittest.TestCase):\n"
            "    def test_ok(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        result = get_backend().run(Path(temp), (sys.executable, "-m", "unittest", "discover", "-q"))
    return Check("sandbox", result.passed, "executes tests" if result.passed else result.output)


def run_checks() -> list[Check]:
    """Run the checks, skipping those whose prerequisites already failed.

    The nesting is deliberate: asking the model to generate when the server is unreachable
    would report a second, derived failure and bury the real one. Each check runs only when the
    thing it depends on passed. The sandbox check is outside that chain because it needs no
    model at all.
    """
    config = LLMConfig.from_env()
    checks = [_check_python(), _check_ram(), _check_ollama_installed(), _check_server(config)]

    if checks[-1].ok:  # server reachable
        checks.append(_check_model_present(config))
        if checks[-1].ok:  # model present
            # Generation must run before the context check: /api/ps can only report the context
            # length of a model that is currently loaded, and generating is what loads it.
            generation = _check_generation(config)
            checks.append(generation)
            if generation.ok:
                # Both of these depend on the model actually answering, so both are gated on
                # generation rather than merely on the model being present. Without the gate a
                # single dead server produced FOUR failures — generation, context, reasoning,
                # tool calling — three of them derived, each with a different and misleading
                # fix attached, and two of them paying for a real model call first.
                checks.append(_check_context(config))
                checks.extend(_check_reasoning_and_tools(config))

    checks.append(_check_sandbox())
    return checks


def report(checks: list[Check]) -> int:
    """Print every check, then a single verdict line. Returns a process exit code."""
    for check in checks:
        mark = "PASS" if check.ok else "FAIL"
        print(f"[{mark}] {check.name}: {check.detail}")

    failed = [check for check in checks if not check.ok]
    if failed:
        print(f"\nNOT READY — {len(failed)} check(s) failed. Fix the FAIL lines above.")
        return 1

    rate = next((c.detail for c in checks if c.name == "generation"), "")
    print(f"\nREADY {rate}".rstrip())
    return 0
