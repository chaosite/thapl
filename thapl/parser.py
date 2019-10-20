#!/usr/bin/env python3

import copy
import logging
import os.path
import typing as t
from abc import ABC, ABCMeta, abstractmethod
from functools import partial, wraps
from itertools import repeat, chain

import lark
import lark.exceptions
import numeral
from lark import Lark, Transformer, Token, v_args
from lark.indenter import Indenter
from lark.visitors import Transformer_InPlaceRecursive

from thapl import then_meanwhile_tree, context, directives
from thapl.exceptions import LookupException
from thapl.number_parser import EnglishNumberParser
from thapl.pattern import Pattern, ParameterizedPattern
from thapl.utility import one_or_raise

GRAMMAR_FILENAME = "thapl.lark"

logging.basicConfig(level=logging.ERROR)


class WhitespacePostlex(Indenter):
    NL_type = "_NL"
    OPEN_PAREN_types: t.Sequence[str] = []
    CLOSE_PAREN_types: t.Sequence[str] = []
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4


class PostlexChain:
    def __init__(self, *postlexers: t.Any) -> None:
        self.postlexers = postlexers

    def process(self, stream: t.Any) -> t.Any:
        for postlexer in self.postlexers:
            stream = postlexer.process(stream)
        return stream

    @property
    def always_accept(self) -> t.Sequence[Token]:
        s: t.Set[Token] = set()
        for postlexer in self.postlexers:
            s |= set(postlexer.always_accept)
        return tuple(s)


class NumberLiteralPostlex:
    def __init__(self) -> None:
        self._english_parser = EnglishNumberParser()

    def process(self, stream: t.Iterator[lark.Token]) -> t.Iterator[Token]:
        number_tokens: t.Optional[t.List[lark.Token]] = None

        for token in stream:
            try:
                if not number_tokens:
                    number_tokens = []
                self._english_parser.parse(number_tokens + [token])
                number_tokens.append(token)
                continue
            except ValueError:
                if number_tokens:
                    v = self._english_parser.parse(number_tokens)
                    yield Token.new_borrow_pos(
                        "UNSIGNED_INTEGER"
                        if isinstance(v, int) else "UNSIGNED_REAL",
                        str(self._english_parser.parse(number_tokens)),
                        number_tokens[0])
                    number_tokens = None
            if not number_tokens:
                try:
                    # Roman numerals must be at least 2 characters long,
                    # otherwise we can't use 'i' or 'x'...
                    if not len(str(token).strip()) < 2:
                        yield Token.new_borrow_pos(
                            "INTEGER", str(numeral.roman2int(token)), token)
                        continue
                except ValueError:
                    pass
                except NotImplementedError:
                    pass
            yield token
        # No need to take care of number_tokens here because there's always an
        # "END" token.

    @property
    def always_accept(self) -> t.Tuple[()]:
        return ()


def open_parser(start: str = "start") -> Lark:
    return Lark.open(GRAMMAR_FILENAME,
                     rel_to=__file__,
                     parser="lalr",
                     start=start,
                     debug=True,
                     postlex=PostlexChain(WhitespacePostlex(),
                                          NumberLiteralPostlex()))


class ArrayPreTransformer(Transformer_InPlaceRecursive):
    DEPTH_MAP = {1: "i", 2: "j", 3: "k", 4: "l", 5: "m", 6: "n"}

    def __init__(self) -> None:
        self.depth = 0

    def _create_array(self, template: lark.Tree, header: lark.Tree,
                      indices: t.Sequence[int]) -> lark.Tree:
        actor_header = lark.Tree("actor_header", [header.children[0]])
        items = [
            self._create_item_at_index(template, index) for index in indices
        ]
        items.insert(0, actor_header)
        items.insert(1, Token("COLON", ":"))
        items.append(lark.Tree("actor_footer", [True]))
        return lark.Tree("actor_def", items)

    def _create_item_at_index(self, template: lark.Tree,
                              index: int) -> lark.Tree:
        item = copy.deepcopy(template)
        identifier = [
            Token("IDENTIFIER", "item"),
            Token("INTEGER", str(index))
        ]
        identifier_tree = lark.Tree("tokens_identifier", identifier)
        actor_header = lark.Tree("actor_header", [identifier_tree])
        item.children[0] = actor_header
        try:
            loc = item.children.index(Token("COLON", ":")) + 1
        except ValueError:
            item.children.insert(-1, Token("COLON", ":"))
            loc = -1
        item.children.insert(loc, self._create_index_property(index))
        return lark.Tree("sub_actor", [item])

    def _create_index_property(self, index: int) -> lark.Tree:
        return lark.Tree("has_property", [
            lark.Tree("has_or_has_meta", [Token("HAS", 'has')]),
            lark.Tree("tokens_identifier",
                      [Token("IDENTIFIER", self.DEPTH_MAP[self.depth])]),
            lark.Tree("single_inheritance", [
                lark.Tree("tokens_identifier",
                          [Token("IDENTIFIER", "integer")])
            ]),
            lark.Tree(
                "initializer",
                [lark.Tree("tokens", [Token("UNSIGNED_INTEGER", str(index))])])
        ])

    @v_args(tree=True)
    def actor_header(self, tree: lark.Tree) -> t.Tuple[bool, lark.Tree]:
        return False, tree

    @v_args(tree=True)
    def actor_header_array(self, tree: lark.Tree) -> t.Tuple[bool, lark.Tree]:
        self.depth += 1
        return True, tree

    @v_args(tree=True)
    def actor_def(self, tree: lark.Tree) -> lark.Tree:
        if tree.children[0][0] is False:
            tree.children[0] = tree.children[0][1]
            return tree
        header = tree.children[0][1]
        indices = header.children[1]
        ret = self._create_array(tree, header, indices)
        self.depth -= 1
        return ret

    @v_args(tree=True)
    def actor_def_with_inheritance(self, tree: lark.Tree) -> lark.Tree:
        return self.actor_def(tree)

    @v_args(inline=True)
    def number_spec(self, *number_ranges: t.Sequence[int]) -> t.Sequence[int]:
        return [i for number_range in number_ranges for i in number_range]

    @v_args(inline=True)
    def number_range(self, start: int,
                     stop: t.Optional[int] = None) -> t.Sequence[int]:
        start = int(start)
        if stop is None:
            stop = start + 1
        else:
            stop = int(stop) + 1
        return range(start, stop)


class TopLevelParser(Transformer):
    """ The top-level parser that parses the basic structure of Thapl. """
    def __init__(self, cwd: t.Optional[str] = None) -> None:
        super().__init__()
        if cwd is None:
            cwd = os.getcwd()
        self._cwd = cwd
        self._parser = open_parser()
        self._array_pre_transformer = ArrayPreTransformer()
        self._nest_depth = 0

    def parse(self, text: str) -> lark.Tree:
        self.__current = [context.Compound()]
        self._sections: t.List[context.Compound] = []
        parse_tree = self._parser.parse(text)
        self._array_pre_transformer.transform(parse_tree)
        try:
            ret = self.transform(parse_tree)
        except lark.exceptions.VisitError as e:
            raise e.orig_exc
        assert len(self.__current) == 1
        return ret

    @property
    def _current(self) -> 'context.Compound':
        return self.__current[-1]

    def lex(self, text):
        return self._parser.lex(text)

    # Generic transformations
    def tokens(self, items):
        return items

    def tokens_identifier(self, items):
        return items

    def tokens_identifier_no_to(self, items):
        return items

    # Transformations for expressions
    @v_args(inline=True)
    def expression(self, expression):
        return context.Reference((expression, ), self._current.value)

    @v_args(inline=True)
    def expr_var(self, identifier):
        var = parse_literal(identifier, self._current.value)
        var.set_do_not_collapse()
        return ReferenceExpression(var)

    @v_args(inline=True)
    def expr_value(self, value):
        value = parse_literal([value], self._current.value)
        return AtomExpression(value)

    @v_args(inline=True)
    def expr_gt(self, left, right):
        return ComparatorExpression(left, "gt", right)

    @v_args(inline=True)
    def expr_lt(self, left, right):
        return ComparatorExpression(left, "lt", right)

    @v_args(inline=True)
    def expr_ge(self, left, right):
        return ComparatorExpression(left, "ge", right)

    @v_args(inline=True)
    def expr_le(self, left, right):
        return ComparatorExpression(left, "le", right)

    @v_args(inline=True)
    def expr_eq(self, left, right):
        return ComparatorExpression(left, "eq", right)

    @v_args(inline=True)
    def expr_ne(self, left, right):
        return ComparatorExpression(left, "ne", right)

    @v_args(inline=True)
    def expr_or(self, left, right):
        return BooleanExpression(left, "_or", right)

    @v_args(inline=True)
    def expr_and(self, left, right):
        return BooleanExpression(left, "_and", right)

    @v_args(inline=True)
    def expr_not(self, expr):
        return BooleanExpression(expr, "_not", None)

    @v_args(inline=True)
    def expr_inline(self, expr):
        return expr

    @v_args(inline=True)
    def expr_neg(self, atom):
        return BinaryExpression(AtomExpression(context.Integer(-1)), "mul",
                                atom)

    @v_args(inline=True)
    def expr_add(self, left, right):
        return BinaryExpression(left, "add", right)

    @v_args(inline=True)
    def expr_sub(self, left, right):
        return BinaryExpression(left, "sub", right)

    @v_args(inline=True)
    def expr_mul(self, left, right):
        return BinaryExpression(left, "mul", right)

    @v_args(inline=True)
    def expr_div(self, left, right):
        return BinaryExpression(left, "div", right)

    @v_args(inline=True)
    def expr_mod(self, left, right):
        return BinaryExpression(left, "mod", right)

    @v_args(inline=True)
    def expr_cond(self, condition, true, false):
        return TernaryExpression(condition, true, false)

    def _construct_fqn(self, name):
        return list(
            chain.from_iterable(
                chain.from_iterable(
                    zip(
                        chain((x._pattern._tokens for x in self.__current[1:]),
                              name), repeat((Token("SCOPE_OPERATOR",
                                                   "\\"), ))))))[:-1]

    # Transformations for structure
    def _generic_header(self, name, **kwargs):
        self.__current.append(
            context.Named(
                Pattern(name),
                context.Compound(name=name,
                                 fqn=self._construct_fqn(name),
                                 **kwargs)))
        return self._current

    def _generic_footer(self) -> None:
        self.__current.pop()

    @v_args(inline=True)
    def section_header(self, name, **kwargs):
        named = self._generic_header(name, **kwargs)
        self._sections.append(named)
        return named

    @v_args(inline=True)
    def section_footer(self):
        self._sections.pop()
        return self._generic_footer()

    @v_args(inline=True)
    def scene(self, obj, body):
        return obj

    @v_args(inline=True)
    def act(self, obj, body, *scenes):
        return obj

    @v_args(inline=True)
    def play(self, obj, body, *acts):
        return obj

    @v_args(inline=True)
    def load_list(self, *items):
        return items

    @v_args(inline=True)
    def load_directive(self, to_load, filename):
        filename = parse_literal([filename], self._current.value).stringify(
            self._current, ())
        if not os.path.isabs(filename):
            filename = os.path.join(self._cwd, filename)
        # TODO: Cache
        with open(filename, "r") as f:
            parser = TopLevelParser(self._cwd)
            obj = parser.parse(f.read())
        for importable in to_load:
            importing = obj
            for name in split_by_scope_operator(importable):
                importing, _ = one_or_raise(importing.find_character(name),
                                            LookupException)
                basename = name
                assert importing is not None
            # Clone to support caching in the future
            importing = importing.clone(scope=self._current.value, first=True)
            self._current.value.append_character(
                context.Named(Pattern(basename), importing))
            # self._current.value.append_character(importing)

    def dramatis_personae(self, children):
        # Do nothing
        pass

    def actors_header(self, children):
        self._actor_type = "actor"

    def scenery(self, children):
        pass

    def scenery_header(self, children):
        self._actor_type = "character"

    def action(self, children):
        self._current.value.command = then_meanwhile_tree.Then(children)

    # Transformations for directives
    @v_args(inline=True)
    def then_directive(self, left, keyword, right):
        return then_meanwhile_tree.Then([left, right])

    @v_args(inline=True)
    def meanwhile_directive(self, left, keyword, right):
        return then_meanwhile_tree.Meanwhile([left, right])

    @v_args(inline=True)
    def single_compound_directive(self, directive):
        return directive

    def call_directive(self, tokens):
        return directives.Call.from_tokens(tokens)

    def change_directive(self, tokens):
        return directives.Change.from_tokens(tokens)

    @v_args(inline=True)
    def atomic_change_slides(self, tokens=()):
        return tokens

    @v_args(inline=True)
    def atomic_change_k(self, tokens=()):
        return tokens

    def set_directive(self, tokens):
        return directives.Set.from_tokens(tokens)

    def relax_directive(self, tokens):
        return directives.Relax.from_tokens(tokens)

    def sentence_directive(self, items):
        return directives.Sentence.from_tokens(items[0])

    def directives(self, items):
        return items

    # Transformations for Thaplon
    def _clone_for_inheritance(
            self,
            parent_name: t.Sequence[Token],
            ctor_call: t.Optional[t.Sequence[t.Any]],
            overrides: t.Sequence[
                t.Tuple[t.Sequence[Token], 'context.Atomic']],
            new_obj: t.Optional['context.Compound'] = None
    ) -> 'context.Compound':
        parent, _ = one_or_raise(
            self._current.value.find_character(parent_name),
            LookupException("Can't find name {0!r}, ({1:d}:{2:d})".format(
                parent_name, parent_name[0].line, parent_name[0].column)))
        if parent is None:
            raise NotImplementedError("Type not found error should go here")
        copy = parent.clone(scope=self.__current[-2],
                            first=True,
                            inheritor=new_obj)
        copy._actor = False  # Mark it as not an actor (scopes *are* actors?)
        for override_name, override_value in overrides:
            obj, _ = next(copy.parse_full_match(override_name), (
                None,
                None,
            ))
            obj.set(override_value)
        if ctor_call is not None:
            copy.constructor_call(ctor_call, token=parent_name[0])
        return copy

    def parse_type(self, type_name: t.Sequence[Token], type_ctor_call: t.Any,
                   type_overrides: t.Any,
                   value: t.Optional['context.Atomic']) -> 'context.Compound':
        if len(type_name) == 1:
            for atomic_type_name, atomic_type in context.Atomic.VALUES:
                if type_name[0].upper() == atomic_type_name.upper():
                    assert type_ctor_call is None
                    if len(type_overrides) != 0:
                        raise NotImplementedError(
                            "Overrides not supported for primitives")
                    if isinstance(value, context.Reference):
                        atomic = value
                        atomic._cast = atomic_type
                    else:
                        atomic = atomic_type(None)
                        if value is not None:
                            atomic.set(value)
                    return atomic
        assert value is None
        ret = self._clone_for_inheritance(type_name, type_ctor_call,
                                          type_overrides)
        ret.append_actor_inherited(context.EnvironmentLink(
            self._current.value))
        return ret

    @v_args(inline=True)
    def has_property(self, is_meta, name, _type, value=None):
        ret = None
        type_name, type_ctor_call, type_overrides = _type
        ret = context.Named(Pattern(name),
                            self.parse_type(type_name, type_ctor_call,
                                            type_overrides, value),
                            is_meta=is_meta)
        self._current.value.append_actor(ret)
        return ret

    @v_args(inline=True)
    def has_property_inferrence(self, is_meta, name, value):
        ret = context.Named(Pattern(name), value, is_meta=is_meta)
        self._current.value.append_actor(ret)
        return ret

    @v_args(inline=True)
    def has_or_has_meta(self, has_keyword, meta_keyword=None):
        assert has_keyword.lower() == "has"
        if meta_keyword is not None:
            assert meta_keyword.lower() == "meta"
        return meta_keyword is not None

    @v_args(inline=True)
    def can_property(self, keyword, name, _colon, variables, modifiers,
                     *directives):
        c = context.Compound(then_meanwhile_tree.Then(directives))
        for pattern, modifier in modifiers:
            c.append_actor(context.Named(pattern, modifier))
        for pattern, type_ in variables:
            c.append_actor(context.VariableLink(context.Named(pattern, type_)))
        ret = context.CanLink(context.Named(Pattern(name), c))
        self._current.value.append_actor(ret)
        return ret

    @v_args(inline=True)
    def sub_actor(self, named):
        return named

    @v_args(inline=True)
    def initializer(self, tokens):
        value = parse_literal(tokens, self._current.value)
        while isinstance(value, context.Reference):
            value = value.value
        return value

    @v_args(inline=True)
    def initializer_lazy(self, tokens):
        return parse_literal(tokens, self._current.value)

    @v_args(inline=True)
    def modifier(self, pattern):
        return ParameterizedPattern(pattern), context.Unit()

    @v_args(inline=True)
    def modifier_with_directives(self, pattern, colon, *directives):
        return (ParameterizedPattern(pattern),
                context.Compound(command=then_meanwhile_tree.Then(directives)))

    def modifiers(self, items):
        return items

    def can_variables(self, items):
        return items

    @v_args(inline=True)
    def can_variable(self, name, tokens):
        return (Pattern(name), parse_literal(tokens, self._current.value))

    @v_args(inline=True)
    def actor_header(self, name):
        named = self._generic_header(name, actor=True)
        self._add_actor(named)
        named.value.append_actor_inherited(
            context.EnvironmentLink(self.__current[-2].value))
        self._nest_depth += 1
        return named

    @v_args(inline=True)
    def actor_footer(self, synthetic=False):
        if synthetic:
            self._current.value.enable_synthetic()
        self._nest_depth -= 1
        return self._generic_footer()

    def _add_actor(self, named):
        if self._actor_type == "actor" or self._nest_depth > 0:
            self.__current[-2].value.insert_actor(named, 0)
            if self._nest_depth > 0:
                self.__current[-2].value.enable_aggregate()
        elif self._actor_type == "character":
            self.__current[-2].value.append_character(named)
        else:
            assert False

    @v_args(inline=True)
    def actor_def(self, named, _colon, *properties):
        return named

    @v_args(inline=True)
    def actor_def_with_inheritance(
            self,
            named: 'context.Named',
            inheritances: t.Sequence[
                t.Tuple[t.Sequence[Token], t.Any, t.Sequence[
                    t.Tuple[t.Sequence[Token], 'context.Atomic']]]],
            _colon=None,
            *properties):
        for parent_name, ctor_call, overrides in inheritances:
            parent = context.InheritanceLink(
                self._clone_for_inheritance(parent_name,
                                            ctor_call,
                                            overrides,
                                            new_obj=named.value))
            # We're purposefully using append_actor here, since we might have
            # previously appended the nesting actor, and the inheritance should
            # come first.
            named.value.append_actor(parent)
        return named

    @v_args(inline=True)
    def constructor(self, *params):
        self._current.value.set_constructor(params)

    @v_args(inline=True)
    def constructor_param(self, name, type_):
        return (Pattern(name), self.parse_type(type_, None, [], None))

    @v_args(inline=True)
    def constructor_call(self, *arguments: t.Sequence[lark.Token]
                         ) -> t.Sequence['context.Atomic']:
        return [parse_literal(arg, self._current.value) for arg in arguments]

    @v_args(inline=True)
    def inheritance(self, *singles):
        return singles

    @v_args(inline=True)
    def single_inheritance(self, name, ctor_call=None, *overrides):
        assert not overrides or overrides[0].type in ("COLON", "BUT")
        return name, ctor_call, overrides[1:]

    def inheritance_property(self, items):
        return (
            items[0],
            items[1],
        )

    @v_args(inline=True)
    def thaplon(self, *actors):
        return actors


def split_by_scope_operator(tokens):
    if len(tokens) == 0:
        return []
    ret = [[]]
    for token in tokens:
        if token.type == "SCOPE_OPERATOR":
            ret.append([])
        else:
            ret[-1].append(token)
    return ret


def parse_literal(value_tokens: t.Sequence[lark.Token],
                  current: 'context.Compound',
                  value_type: t.Type['context.Obj'] = None) -> 'context.Obj':
    # TODO: Move this elsewhere? add `assert` to make sure it's complete?
    atomics = {
        "SHORT_STRING": context.String,
        "LONG_STRING": context.String,
        "INTEGER": context.Integer,
        "UNSIGNED_INTEGER": context.Integer,
        "UNIT": context.Unit,
        "REAL": context.Real,
        "UNSIGNED_REAL": context.Real,
        "BOOLEAN": context.Boolean,
        "REFERENCE": partial(context.Reference, scope=current),
        "VARIABLE": partial(context.Reference, scope=current),
        "FLAG": context.Flag
    }

    # if it is actually already parsed
    if value_tokens is not None and len(value_tokens) == 1 and \
       isinstance(value_tokens[0],
                  tuple([value[1] for value in context.Atomic.VALUES])):
        if value_type is not None:
            if value_type == context.Reference:
                value = value_type(None, scope=current)
            else:
                value = value_type(None)
            value.set(value_tokens[0])
            return value
        return value_tokens[0]

    if value_tokens is not None and (len(value_tokens) > 1
                                     or value_tokens[0].type == "IDENTIFIER"):
        # Check to see if this is referencing a ctor parameter
        ctor_value = current.constructor_param(value_tokens)
        if ctor_value is not None:
            return context.Reference(
                (ConstructorParamExpression(current, value_tokens), ),
                scope=current)

        # Otherwise, treat it as a reference
        return context.Reference(value_tokens, scope=current)
    assert value_tokens is None or len(value_tokens) == 1

    if value_type is None:
        # Try to infer the value type and use it.
        assert value_tokens is not None and value_tokens[0].type in atomics
        value = atomics[value_tokens[0].type](None)
        value.parse_literal(value_tokens)
        return value
    else:
        # We have a type specification, use it.
        assert hasattr(value_type, 'parse_literal')
        if value_type == context.Reference:
            value = value_type(None, scope=current)
        else:
            value = value_type(None)
        if value_tokens is not None:
            value.parse_literal(value_tokens)
        return value


def promote(
        f: t.
        Callable[[t.Any, 'context.Atomic', 'context.Atomic'], 'context.Obj']):
    """ Decorator to promote Integers to Reals in expressions. """
    @wraps(f)
    def func(self, left: 'Expression',
             right: 'Expression') -> 'context.Boolean':
        if (issubclass(left.type, context.Real)
                or issubclass(right.type, context.Real)
                and not issubclass(left.type, right.type)):
            if issubclass(left.type, context.Integer):
                left = AtomExpression(context.Real(float(left.value.value)))
            if issubclass(right.type, context.Integer):
                right = AtomExpression(context.Real(float(right.value.value)))
        return f(self, left, right)

    return func


class Expression(ABC):
    @property
    @abstractmethod
    def value(self) -> 'context.Obj':
        pass

    @property
    @abstractmethod
    def type(self) -> t.Type['context.Obj']:
        pass

    @abstractmethod
    def clone(self, **kwargs: t.Any) -> 'Expression':
        pass


class ProtoBinaryExpression(Expression, metaclass=ABCMeta):
    def __init__(self, left: Expression, op: str,
                 right: t.Optional[Expression]) -> None:
        self._left = left
        self._op = op
        self._right = right

    @property
    def value(self) -> 'context.Atomic':
        return getattr(self, self._op)(self._left, self._right)

    @property
    @abstractmethod
    def type(self) -> t.Type['context.Atomic']:
        raise NotImplementedError()

    def clone(self, **kwargs) -> Expression:
        def clone(thing):
            if thing is None:
                return None
            return thing.clone(**kwargs)

        return self.__class__(clone(self._left), self._op, clone(self._right))


class BinaryExpression(ProtoBinaryExpression):
    @property
    def type(self) -> t.Type['context.Atomic']:
        ltype = self._left.type
        assert self._right is not None
        rtype = self._right.type
        if ltype != rtype:
            assert ltype in (context.Real, context.Integer, context.String)
            assert rtype in (context.Real, context.Integer, context.String)
            return context.Real
        return ltype

    @promote
    def add(self, left: 'context.Atomic',
            right: 'context.Atomic') -> 'context.Atomic':
        return left.value + right.value

    @promote
    def sub(self, left: 'context.Atomic',
            right: 'context.Atomic') -> 'context.Atomic':
        return left.value - right.value

    @promote
    def mul(self, left: 'context.Atomic',
            right: 'context.Atomic') -> 'context.Atomic':
        return left.value * right.value

    @promote
    def div(self, left: 'context.Atomic',
            right: 'context.Atomic') -> 'context.Atomic':
        return left.value / right.value

    @promote
    def mod(self, left: 'context.Atomic',
            right: 'context.Atomic') -> 'context.Atomic':
        return left.value % right.value


class ComparatorExpression(ProtoBinaryExpression):
    @property
    def type(self) -> t.Type['context.Boolean']:
        return context.Boolean

    @promote
    def gt(self, left: 'context.Atomic',
           right: 'context.Atomic') -> 'context.Boolean':
        return left.value > right.value

    @promote
    def lt(self, left: 'context.Atomic',
           right: 'context.Atomic') -> 'context.Boolean':
        return left.value < right.value

    @promote
    def ge(self, left: 'context.Atomic',
           right: 'context.Atomic') -> 'context.Boolean':
        return left.value >= right.value

    @promote
    def le(self, left: 'context.Atomic',
           right: 'context.Atomic') -> 'context.Boolean':
        return left.value <= right.value

    @promote
    def eq(self, left: 'context.Atomic',
           right: 'context.Atomic') -> 'context.Boolean':
        return left.value == right.value

    @promote
    def ne(self, left: 'context.Atomic',
           right: 'context.Atomic') -> 'context.Boolean':
        return left.value == right.value


class BooleanExpression(ProtoBinaryExpression):
    @property
    def type(self) -> t.Type['context.Boolean']:
        return context.Boolean

    def _or(self, left: Expression, right: Expression) -> 'context.Boolean':
        return left.value or right.value

    def _and(self, left: Expression, right: Expression) -> 'context.Boolean':
        return left.value and right.value

    def _not(self, left: Expression, right: None = None) -> 'context.Boolean':
        assert right is None
        return context.Boolean(not left.value)

    def clone(self, *args: t.Any, **kwargs: t.Any) -> 'BooleanExpression':
        return t.cast('BooleanExpression', super().clone(*args, **kwargs))


class TernaryExpression(Expression):
    def __init__(self, condition: BooleanExpression, true: Expression,
                 false: Expression) -> None:
        self._cond = condition
        self._true = true
        self._false = false

    @property
    def value(self) -> t.Optional['context.Obj']:
        if self._cond.type is None:
            return None
        assert self._cond.type is context.Boolean
        assert self._cond.value.value is not None
        if self._cond.value.value:
            return self._true.value
        else:
            return self._false.value

    @property
    def type(self) -> t.Type['context.Obj']:
        assert self._cond.type is context.Boolean
        ttype = self._true.type
        ftype = self._false.type
        if ttype != ftype:
            assert ttype in (
                context.Real,
                context.Integer,
            )
            assert ftype in (
                context.Real,
                context.Integer,
            )
            return context.Real
        return ttype

    def clone(self, **kwargs: t.Any) -> 'TernaryExpression':
        return self.__class__(self._cond.clone(**kwargs),
                              self._true.clone(**kwargs),
                              self._false.clone(**kwargs))


class AtomExpression(Expression):
    def __init__(self, number: 'context.Atomic') -> None:
        self._number = number

    @property
    def value(self) -> 'context.Atomic':
        return self._number

    @property
    def type(self) -> t.Type['context.Atomic']:
        return self._number.type

    def clone(self, **kwargs: t.Any) -> 'AtomExpression':
        return self  # immutable


class ConstructorParamExpression(Expression):
    def __init__(self, current: 'context.Compound',
                 param: t.Sequence[lark.Token]) -> None:
        super().__init__()
        self._current = current
        self._param = param
        self._type_cache: t.Optional[t.Type['context.Obj']] = None

    @property
    def value(self) -> 'context.Obj':
        return self._current.constructor_param(self._param)

    @property
    def type(self) -> t.Type['context.Obj']:
        if self._type_cache is None:
            self._type_cache = self._current.constructor_param(self._param)
        return self._type_cache

    def clone(self,
              replacements: t.Optional[t.Dict[int, 'context.Obj']] = None,
              **kwargs: t.Any) -> 'ConstructorParamExpression':
        new = self._current
        if replacements is not None and id(self._current) in replacements:
            new = replacements[id(self._current)]
        return self.__class__(new, self._param)


class ReferenceExpression(Expression):
    def __init__(self, reference: 'context.Reference') -> None:
        # assert not isinstance(reference.value, context.Reference)
        super().__init__()
        assert reference._do_not_collapse
        self._reference = reference
        self._type_cache: t.Optional[t.Type['context.Obj']] = None

    @property
    def value(self) -> 'context.Obj':
        return self._reference.value

    @property
    def type(self) -> t.Type['context.Obj']:
        if self._type_cache is None:
            self._type_cache = self._reference.type
        return self._type_cache

    def clone(self, **kwargs: t.Any) -> 'ReferenceExpression':
        return self.__class__(self._reference.clone(**kwargs))
