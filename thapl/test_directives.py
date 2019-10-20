import unittest

from thapl import directives, context


class TestRelax(unittest.TestCase):
    def test_init(self):
        directives.Relax()


class TestCall(unittest.TestCase):
    def test_init(self):
        directives.Call(None)
        directives.Call.from_tokens(["call", "me", "maybe"])
