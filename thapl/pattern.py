#!/usr/bin/env python3
"""
Pattern-matching is used in Thapl in order to recognize identifiers.
"""
from thapl.nfa import NFA


class Pattern:
    """
    Class representing a pattern.
    """

    def __init__(self, tokens):
        self._tokens = tokens

    def match(self, tokens):
        """
        Match tokens to pattern, returning a 3-tuple containing the remaining
        unmatched tokens, a boolean value of whether there was a successful
        match, and any metadata about the match.

        Checks for strict equality to the pattern tokens from the beginning of
        list.

        :param tokens: Token list to match against this pattern.

        """
        if len(self._tokens) > len(tokens):
            return tokens, False, None

        if not all(a == b for a, b in zip(tokens, self._tokens)):
            return tokens, False, None

        return tokens[len(self._tokens):], True, None

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self._tokens)


class ParameterizedPattern(Pattern):
    """
    Class representing a pattern with parameters.
    """

    def __init__(self, tokens):
        super().__init__(tokens)
        assert len(tokens) != 0
        self.nfa = NFA(self._tokens)

    def match(self, tokens):
        m = self.nfa.match(tokens)
        if m is None:
            return tokens, False, None
        return [], True, m
