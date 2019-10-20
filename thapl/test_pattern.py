import unittest

from thapl import pattern
from thapl.test_utils import tokenize


class TestParameterizedPattern(unittest.TestCase):
    def test_init(self):
        pattern.ParameterizedPattern(tokenize("foo $var bar"))

    def test_match1(self):
        p = pattern.ParameterizedPattern(tokenize("foo $var bar"))
        rest, s, v = p.match(tokenize("foo buzz blarg bar"))
        self.assertEqual(0, len(rest))
        self.assertTrue(s)
        self.assertEqual(list(tokenize("buzz blarg")),
                         v[list(tokenize("$var"))[0]])
