import unittest

from thapl import nfa
from thapl.test_utils import tokenize


class TestNfa(unittest.TestCase):
    def test_init(self):
        nfa.NFA(tokenize("no wildcards"))

    def test_init_with_variables(self):
        nfa.NFA(tokenize("something $variable"))

    def test_init_with_variables2(self):
        nfa.NFA(tokenize("foo $var bar"))

    def test_match1(self):
        m = nfa.NFA(tokenize("something $variable"))
        ret = m.match(tokenize("something or other"))
        result = ret[list(tokenize("$variable"))[0]]
        self.assertEqual(list(tokenize("or other")), result)

    def test_match2(self):
        m = nfa.NFA(tokenize("this is $variable and that is $variable2"))
        ret = m.match(
            tokenize("this is madness really and that is not sparta for real"))
        self.assertIsNotNone(ret)
        self.assertEqual(
            list(tokenize("madness really")),
            ret[list(tokenize("$variable"))[0]])
        self.assertEqual(
            list(tokenize("not sparta for real")), ret[list(
                tokenize("$variable2"))[0]])

    def test_match3(self):
        m = nfa.NFA(tokenize("this is $variable and that is $variable2 too"))
        ret = m.match(
            tokenize(
                "this is madness really and that is not sparta for real too"))
        self.assertIsNotNone(ret)
        self.assertEqual(
            list(tokenize("madness really")),
            ret[list(tokenize("$variable"))[0]])
        self.assertEqual(
            list(tokenize("not sparta for real")), ret[list(
                tokenize("$variable2"))[0]])

    def test_match4(self):
        m = nfa.NFA(tokenize("just some literals"))
        self.assertIsNotNone(m.match(tokenize("just some literals")))

    def test_nomatch(self):
        m = nfa.NFA(tokenize("this $variable"))
        self.assertIsNone(m.match(tokenize("that is wrong")))
        self.assertIsNone(m.match(tokenize("this")))
        self.assertIsNone(m.match(tokenize("")))

    def test_one_any(self):
        m = nfa.NFA(tokenize("$any"))
        self.assertEqual(
            list(tokenize("foo foo foo")),
            m.match(tokenize("foo foo foo"))[tuple(tokenize("$any"))[0]])
