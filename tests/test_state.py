"""The state's reducers: how a node's partial update combines with what is already there."""

from __future__ import annotations

import inspect
import unittest

from langchain_core.messages import AIMessage

from agentfix.agent.graph import completion_tokens_of
from agentfix.agent.state import initial_state, keep_larger
from agentfix.agent.trace import prompt_tokens_of


class TestKeepLarger(unittest.TestCase):
    def test_the_larger_value_wins_in_both_directions(self):
        self.assertEqual(keep_larger(5, 3), 5)
        self.assertEqual(keep_larger(3, 5), 5)
        self.assertEqual(keep_larger(4, 4), 4)

    def test_a_high_water_mark_starts_correctly_from_zero(self):
        self.assertEqual(keep_larger(0, 120), 120)
        self.assertEqual(initial_state("s", "t")["peak_prompt_tokens"], 0)

    def test_the_builtin_max_cannot_be_used_as_a_reducer(self):
        """The reason this function exists, asserted so nobody "simplifies" it back.

        LangGraph inspects a reducer's signature to recognise it as a two-argument combiner, and
        `inspect.signature` raises on a C builtin — so `Annotated[int, max]` fails at StateGraph
        construction with "no signature found for builtin max".
        """
        with self.assertRaises(ValueError):
            inspect.signature(max)
        self.assertIsNotNone(inspect.signature(keep_larger))


class TestUsageAccessors(unittest.TestCase):
    def test_a_reply_with_no_usage_metadata_costs_zero_rather_than_raising(self):
        """Not every server reports usage; the accounting must degrade, not crash."""
        bare = AIMessage(content="hello")
        self.assertEqual(prompt_tokens_of(bare), 0)
        self.assertEqual(completion_tokens_of(bare), 0)

    def test_usage_is_read_from_the_metadata_when_present(self):
        message = AIMessage(
            content="hi",
            usage_metadata={"input_tokens": 120, "output_tokens": 7, "total_tokens": 127},
        )
        self.assertEqual(prompt_tokens_of(message), 120)
        self.assertEqual(completion_tokens_of(message), 7)
