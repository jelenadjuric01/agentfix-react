import unittest

from config.parser import parse


class TestParser(unittest.TestCase):
    def test_parse_simple(self):
        self.assertEqual(parse("a=1\nb=2"), {"a": "1", "b": "2"})

    def test_parse_skips_comments(self):
        self.assertEqual(parse("# ignored\na=1"), {"a": "1"})

    def test_parse_trims_whitespace(self):
        self.assertEqual(parse("  a =  1  "), {"a": "1"})
