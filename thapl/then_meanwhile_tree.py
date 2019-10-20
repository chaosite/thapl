#!/usr/bin/env python3
import textwrap
import typing as t

from thapl import spring

if t.TYPE_CHECKING:
    from thapl import log


class ThenMeanwhileTree:
    """
    An abstract Meanwhile/Then tree that holds nothing but
    structure; a typical use is a walk, either bottom up or top down, adding
    baggage fields to nodes.
    """
    def __init__(self) -> None:
        self._children: t.Sequence['ThenMeanwhileTree'] = []

    @property
    def children(self) -> t.Sequence['ThenMeanwhileTree']:
        """ Returns list of children of this node """
        return self._children

    def bottomup(self, visitor: 'Visitor') -> None:
        for child in self.children:
            child.bottomup(visitor)
        self.visit(visitor)

    def topdown(self,
                visitor: 'Visitor',
                parent: t.Optional['ThenMeanwhileTree'] = None,
                left: t.Optional['ThenMeanwhileTree'] = None) -> None:
        self.visit(visitor, parent=parent, left=left)
        for child in self.children:
            child.topdown(visitor, self, left)

    def visit(self,
              visitor: 'Visitor',
              parent: t.Optional['ThenMeanwhileTree'] = None,
              left: t.Optional['ThenMeanwhileTree'] = None) -> None:
        raise NotImplementedError()

    def reduce(self, reducer: 'Reducer') -> 'ThenMeanwhileTree':
        return self.merge(reducer,
                          reduced_children=[
                              child.reduce(reducer) for child in self.children
                          ])

    def merge(self, reducer: 'Reducer',
              reduced_children: t.Sequence['ThenMeanwhileTree']
              ) -> 'ThenMeanwhileTree':
        raise NotImplementedError()


class Compound(ThenMeanwhileTree):
    pass


class Then(Compound):
    def __init__(self, children) -> None:
        self._children = children

    def topdown(self, visitor, parent=None, left=None):
        # TODO: Refactor to reduce code duplication
        self.visit(visitor, parent=parent, left=left)
        left = None
        for child in self.children:
            child.topdown(visitor, self, left)
            left = child

    def visit(self, visitor, **kwargs):
        visitor.then(self, **kwargs)

    def merge(self, reducer, reduced_children):
        return reducer.then(self, reduced_children)


class Meanwhile(Compound):
    def __init__(self, children):
        self._children = children

    def visit(self, visitor, **kwargs):
        visitor.meanwhile(self, **kwargs)

    def merge(self, reducer, reduced_children):
        return reducer.meanwhile(self, reduced_children)


class Atomic(ThenMeanwhileTree):
    def visit(self, visitor, **kwargs):
        visitor.atomic(self, **kwargs)

    def merge(self, reducer, reduced_children=None):
        return reducer.atomic(self)

    def reduce(self, reducer: 'Reducer'):
        return self.merge(reducer)


class Reducer:
    def then(self, tree: Then,
             reduced_children: t.Sequence['ThenMeanwhileTree']
             ) -> ThenMeanwhileTree:
        raise NotImplementedError()

    def meanwhile(self, tree: Meanwhile,
                  reduced_children: t.Sequence['ThenMeanwhileTree']
                  ) -> ThenMeanwhileTree:
        raise NotImplementedError()

    def atomic(self, tree: Atomic) -> ThenMeanwhileTree:
        raise NotImplementedError()


class StringSummarizer(Reducer):
    def __init__(self, _property=None):
        self._property = _property

    def atomic(self, t):
        if self._property is not None and hasattr(t, self._property):
            return repr(getattr(t, self._property))
        return repr(t)

    @staticmethod
    def indent(lines):
        return textwrap.indent("\n".join(lines), " " * 2)

    def property_text(self, t):
        if self._property is None or not hasattr(t, self._property):
            return ""
        return " ({} = {})".format(self._property,
                                   repr(getattr(t, self._property)))

    def then(self, t, texts):
        return "Then{}:\n{}".format(self.property_text(t), self.indent(texts))

    def meanwhile(self, t, texts):
        return "Meanwhile{}:\n{}".format(self.property_text(t),
                                         self.indent(texts))


class Visitor:
    def then(self, tree):
        raise NotImplementedError()

    def meanwhile(self, tree):
        raise NotImplementedError()

    def atomic(self, tree):
        raise NotImplementedError()

    def visit(self, tree):
        raise NotImplementedError()


class BottomupVisitor(Visitor):
    def visit(self, tree):
        tree.bottomup(self)


class TopdownVisitor(Visitor):
    def visit(self, tree):
        tree.topdown(self)


class SpringAttacher(BottomupVisitor):
    def atomic(self, t):
        pass  # Atomic nodes are instructions, and are created with springs

    def then(self, t):
        t.spring = spring.SerialSpring([child.spring for child in t.children])

    def meanwhile(self, t):
        t.spring = spring.ParallelSpring(
            [child.spring for child in t.children])


class TimeSupplier(BottomupVisitor):
    def atomic(self, t):
        if t.mutation is not None:
            t.mutation.supply(t.spring.interval)

    def then(self, *args, **kwargs):
        pass

    def meanwhile(self, *args, **kwargs):
        pass


T = t.TypeVar('T')


class Compacter(Reducer):
    def _compact(self, junction_type: t.Type[T],
                 children: t.Sequence[ThenMeanwhileTree]) -> T:
        if len(children) == 1:
            return children[0]

        new = []
        for child in children:
            if isinstance(child, junction_type):
                new.extend(child.children)
            else:
                new.append(child)
        return junction_type(new)

    def then(self, _: Then, children: t.Sequence[ThenMeanwhileTree]) -> Then:
        return self._compact(Then, children)

    def meanwhile(self, _: Meanwhile,
                  children: t.Sequence[ThenMeanwhileTree]) -> Meanwhile:
        return self._compact(Meanwhile, children)

    def atomic(self, atomic: Atomic) -> Atomic:
        return atomic


class Transformer(Reducer):
    def then(self, _, children):
        return Then(filter(None.__ne__, children))

    def meanwhile(self, _, children):
        return Meanwhile(filter(None.__ne__, children))


class InterpretTransformer(Transformer):
    def __init__(self, context):
        self._context = context

    def atomic(self, directive):
        return directive.interpret(self._context)


class ExecuteTransformer(Transformer):
    def __init__(self, context) -> None:
        self._context = context

    def atomic(self, instruction) -> 'log.Log':
        return instruction.execute(self._context)
