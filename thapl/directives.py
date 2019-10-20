#!/usr/bin/env python3
"""
Internal representation of atomic Thapl directives, without binding information.
- These come in three varieties:
  + Relax, which is the usual NOP,
  + Change, in which a single slot changes its value according to its prorotocol,
  + Sentence, which is a Thapl equivalent of a procedure call.

Compound directives use the ThenMeanwhileTree structure.
"""
import typing as t

import thapl
from thapl import instruction, then_meanwhile_tree, parser
from thapl.exceptions import LookupException
from thapl.utility import one_or_raise, first_or_raise
from thapl.findoptions import FindOptions

PARSER = None


class AtomicDirective(then_meanwhile_tree.Atomic):
    """ Base class for atomic directives """
    _PARSER = None

    def __init__(self):
        global PARSER
        if PARSER is None:
            PARSER = parser.TopLevelParser()
        self._PARSER = PARSER


class Sentence(AtomicDirective):
    """
    A sentence is an atomic directive containing one or more subjects, a single
    verb, and one or more modifiers on those verbs, e.g., in `box_a moves 2
    spaces left.` the subject is box_a, the verb is moves, and the modifiers
    are 2 spaces and left.
    """
    def __init__(self, words: t.Sequence[str]):
        super().__init__()
        self._words = words

    @classmethod
    def from_tokens(cls, tokens: t.Sequence[str]) -> 'Sentence':
        return cls(tokens)

    def interpret(self,
                  obj: 'thapl.context.Obj') -> then_meanwhile_tree.Compound:
        """Parse this sentence using the given object, which is probably a
        compound activity in this program.

        :param obj: An object containing sentences.

        """
        instructions = []

        # Step 1: Parse subject (assume single)
        subject, metadata, subject_rest = one_or_raise(
            obj.value.parse(self._words), LookupException())
        assert subject is not None, "subject not found"
        assert subject_rest, "sentence ended prematurely"

        # Step 1.5: Expand subject to catch nested items:
        for expansion in subject.expand():
            # Step 2: Parse verb (get first)
            try:
                verb, metadata, verb_rest = first_or_raise(
                    expansion.parse(subject_rest), LookupException())
            except LookupException:
                continue

            # Step 3: Collect modifiers and create instruction.
            modifiers = list(self._collect_modifiers(verb_rest, verb, obj))

            instructions.append(
                instruction.Invocation(expansion,
                                       verb,
                                       modifiers,
                                       directive=self))
        if len(instructions) == 0:
            raise LookupException()

        return then_meanwhile_tree.Meanwhile(instructions)

    def _collect_modifiers(
            self, rest: t.Sequence[str], verb: 'thapl.context.Obj',
            obj: 'thapl.context.Obj'
    ) -> t.Iterator[t.Tuple['thapl.context.Obj', t.
                            Dict[t.Sequence[str], 'thapl.context.Atomic']]]:
        orig = rest
        while len(rest) != 0:
            modifier, metadata, rest = min(verb.parse(
                rest, options=FindOptions(search_variables=False)),
                                           key=lambda k: len(k[2]))
            assert modifier is not None, f"Bad modifier error ({orig})"
            yield (modifier, {
                name: parser.parse_literal(value, obj)
                for name, value in metadata.patterns.items()
            })

    def __repr__(self):
        return "Sentence({})".format(self._words)


class Change(AtomicDirective):
    """
    This class represents a "change" sentence, like `change box_a/x to 20`.
    """
    def __init__(self, subject, value, length, k):
        super().__init__()
        self._subject = tuple(subject)
        self._value = tuple(value)
        self._length = tuple(length) if len(length) > 0 else None
        self._k = tuple(k) if len(k) > 0 else None

    @classmethod
    def from_tokens(cls, tokens) -> 'Change':
        """
        Builder class method to create a Change directive from a token list.

        :param tokens: Token list for this directive.

        """
        return cls(tokens[1], tokens[3], tokens[4], tokens[5])

    def interpret(self, obj):
        # Step 1: Find the subject. We expect to find exactly one.
        subject, metadata = one_or_raise(
            obj.value.parse_full_match(self._subject), LookupException())
        assert subject is not None

        # Step 2: Create the instruction.
        value = parser.parse_literal(self._value, obj)
        length = (parser.parse_literal(self._length, obj)
                  if self._length else None)
        k = parser.parse_literal(self._k, obj) if self._k else None
        return instruction.Change(subject,
                                  value,
                                  obj,
                                  length=length,
                                  k=k,
                                  directive=self)

    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__, self._subject, self._value)


class Set(Change):
    def __init__(self, subject, value):
        super().__init__(subject, value, (), ())

    def interpret(self, obj):
        return instruction.Set.from_change(super().interpret(obj))

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[1], tokens[3])


class Call(AtomicDirective):
    """ A call to a sub section """
    def __init__(self, subsection):
        super().__init__()
        self._subsection = subsection

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens[1])

    def interpret(self, obj):
        # Assume there's only one result?
        call = [
            call for todo in obj.find(self._subsection)
            for call in todo.results
        ]
        assert len(call) == 1, call
        closure_obj, directive = call.pop()
        return instruction.Call(directive, closure_obj, directive=self)

    def __repr__(self):
        return "Call({})".format(self._subsection)


class Relax(AtomicDirective):
    """ Atomic directive that does nothing for as long as needed. """
    def interpret(self, obj):
        """
        Parse the relax directive, returning a relax action.

        :param obj: An object to return unmodified.

        """
        return instruction.Relax(directive=self)

    @classmethod
    def from_tokens(cls, tokens):
        return cls()

    def __repr__(self):
        return "Relax()"
