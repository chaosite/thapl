#!/usr/bin/env python3
"""
Context classes
"""

import collections.abc
import typing as t
from abc import ABC, abstractmethod
from functools import wraps

import lark

import thapl.stringexpr as s
from thapl import then_meanwhile_tree, directives, mutations, parser
from thapl.exceptions import LookupException, TypeException, CtorException
from thapl.findoptions import FindMetadata, FindOptions

T = t.TypeVar('T')


class Obj(ABC):
    @abstractmethod
    def parse(self,
              tokens: t.Sequence[str],
              receiver: t.Optional['Obj'] = None,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple['Obj', FindMetadata, t.Sequence[str]]]:
        """Given tokens, find the longest prefix match for this object, and
        return the found object, any associated data for the match, as well as
        the unused tokens.
s
        :param tokens: Unprocessed tokens.

        :returns: Found object, associated match data, and remaining tokens.
        """
        pass

    def parse_full_match(self,
                         tokens: t.Sequence[str],
                         receiver: t.Optional['Obj'] = None,
                         options: FindOptions = FindOptions()
                         ) -> t.Iterator[t.Tuple['Obj', FindMetadata]]:
        if receiver is None:
            receiver = self
        found = False
        for obj, metadata, remaining in self.parse(tokens, receiver, options):
            if len(remaining) == 0:
                found = True
                yield obj, metadata
        if not found:
            raise LookupException("Can't find [{0!r}], ({1:d}:{2:d})".format(
                tokens,
                t.cast(lark.Token, tokens[0]).line,
                t.cast(lark.Token, tokens[0]).column))

    @abstractmethod
    def begin(self) -> then_meanwhile_tree.ThenMeanwhileTree:
        """Return the command associated with this `Obj`, possibly including
        sub-`Objs`.

        :returns: A command representing the actions to execute for this `Obj`.
        """
        pass

    @abstractmethod
    def find_actor(self,
                   name: t.Sequence[str],
                   receiver: t.Optional['Obj'],
                   options: FindOptions = FindOptions()
                   ) -> t.Iterator[t.Tuple['Obj', t.Any]]:
        """Search the `Obj`'s actor list for any sub-`Obj`s with the given
        `name`.

        May return more than one actor.

        :param name: The name to search for.

        :returns: An `Obj` with the requested name, and associated match
                  information.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_character(self,
                       name: t.Sequence[str],
                       receiver: t.Optional['Obj'],
                       options: FindOptions = FindOptions()
                       ) -> t.Iterator[t.Tuple['Obj', t.Any]]:
        """Search the `Obj`'s character list for any sub-`Obj`s with the given
        `name`.

        :param name: The name to search for.

        :returns: An `Obj` with the requested name, and associated match
                  information.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def value(self) -> t.Union['Obj', t.Any]:
        """
        :returns: The value of this `Obj`, or itself if it does not have a
                  discrete value.
        """
        pass

    @property
    @abstractmethod
    def type(self) -> t.Type:
        """Return the type of this Obj.

        :returns: The type of this object.
        """
        pass

    @abstractmethod
    def type_check(self, other: 'Obj') -> bool:
        """Perform type checking, that is, check if another `Obj` can be
        assigned to his `Obj`.

        :param other: The other object to test.

        :returns: `True` if assignment is possible, `False` otherwise.
        """
        pass

    @abstractmethod
    def clone(self, **kwargs) -> 'Obj':
        """Create a deep-copy of this `Obj`.

        :returns: A copy of this `Obj`.
        """
        pass

    @abstractmethod
    def stringify(self, scope: 'Compound', name: t.Sequence[str]):
        """Transform this `Obj` into a string.

        :param scope: A scope to perform string-expression lookups with.
        :param name: The name of this `Obj`, if applicable.

        :returns: A string representation of this `Obj`.
        """
        pass

    def is_reference(self) -> bool:
        """Is this `Obj` a reference?

        :returns: True if a reference, False otherwise.
        """
        return False

    def is_name_shown(self) -> bool:
        """When `stringify`'ing this `Obj`, should the name be shown?

        :returns: True if the name should be shown, False otherwise.
        """
        return True

    def is_atomic(self) -> bool:
        """Is this `Obj` atomic?

        :returns: True if atomic, False otherwise.
        """
        return False

    def expand(self) -> t.Iterator['Obj']:
        """Expand this `Obj` to include any nested sub-`Obj`s, possibly (but not
        necessarily) including this one.

        :returns: Sequence of expanded `Obj`s.
        """
        yield self


def deref_decorator(f):
    """Decorator for automatically dereferncing :class:`Reference` instances to
    the pointed-at value.  Also performs (dynamic) type-checking.
    """
    @wraps(f)
    def wrapper(self, other):
        if isinstance(other, (Reference, parser.Expression)):
            other = other.value
        # TODO: Change assert to custom exception.
        assert self.type_check(other)
        return f(self, other)

    return wrapper


class Atomic(Obj, t.Generic[T]):
    VALUES: t.List[t.Tuple[str, t.Type['Atomic']]] = []
    ALLOWED_ADDITIONAL_TYPES: t.Tuple[t.Type['Atomic'], ...] = ()

    @abstractmethod
    def set_value(self, other: T) -> None:
        """Set the underlying value of this slot to another value, of a matching
        type.  Performs type-checking.

        :param other: The value to assign.
        """
        pass

    @abstractmethod
    def mutation(self, other: 'Atomic[T]') -> mutations.Mutation:
        """Return a mutation for this `Atomic` into the value represented by
        another value.

        :param other: An other value of a matching type.

        :returns: A mutation for this change.
        """
        pass

    @abstractmethod
    def parse_literal(self, tokens: t.Sequence[str]) -> None:
        """Given unparsed tokens representing a literal, assign the literal to
        this slot.

        :param tokens: Unparsed tokens representing a literal.
        """
        pass

    def __init__(self, value: T) -> None:
        super().__init__()
        if value is None:
            self._value = self._default
        else:
            self.set_value(value)

    @property
    def value(self) -> T:
        return self._value

    @property
    def type(self) -> t.Type['Atomic']:
        return self.__class__

    def type_check(self, other: Obj):
        return isinstance(other,
                          (self.__class__, ) + self.ALLOWED_ADDITIONAL_TYPES)

    @deref_decorator
    def set(self, other: 'Atomic[T]'):
        """Assign value to this `Atomic` class, in a `raw` fashion.

        :param other: The value to assign.
        """

        self.set_value(other.value)

    def clone(self, **kwargs: t.Any) -> 'Atomic':
        return self.__class__(self.value)

    def begin(self) -> 'directives.AtomicDirective':
        raise NotImplementedError()

    def find_actor(self, *args, **kwargs) -> Obj:
        raise NotImplementedError()

    def find_character(self, *args, **kwargs) -> Obj:
        raise NotImplementedError()

    def parse(self,
              tokens: t.Sequence[str],
              receiver: t.Optional[Obj] = None,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple[Obj, t.Any, t.Sequence[str]]]:
        raise NotImplementedError()

    @classmethod
    def register_atomic(cls: t.Type['Atomic'], atomic_cls: t.Type['Atomic']):
        cls.VALUES.append((atomic_cls.__name__, atomic_cls))
        return atomic_cls

    @deref_decorator
    def __eq__(self, other: 'Obj') -> 'Boolean':
        return Boolean(self._value == other.value)

    @deref_decorator
    def __ne__(self, other: 'Obj') -> 'Boolean':
        return t.cast('Boolean', not (self == other))

    def is_atomic(self) -> bool:
        return True

    @property
    def _default(self) -> t.Optional[T]:
        return None


class NumericAtomic(Atomic[T], t.Generic[T]):
    @deref_decorator
    def __add__(self, other: 'NumericAtomic[T]') -> 'NumericAtomic[T]':
        assert isinstance(other, self.__class__)
        return self.__class__(other._value + self._value)

    @deref_decorator
    def __sub__(self, other: 'NumericAtomic[T]') -> 'NumericAtomic[T]':
        assert isinstance(other, self.__class__)
        return self.__class__(self._value - other._value)

    @deref_decorator
    def __mul__(self, other: 'NumericAtomic[T]') -> 'NumericAtomic[T]':
        assert isinstance(other, self.__class__)
        return self.__class__(other._value * self._value)

    @deref_decorator
    def __lt__(self, other: 'NumericAtomic[T]') -> 'Boolean':
        return Boolean(self._value < other._value)

    @deref_decorator
    def __gt__(self, other: 'NumericAtomic[T]') -> 'Boolean':
        return Boolean(self._value > other._value)

    @deref_decorator
    def __ge__(self, other: 'NumericAtomic[T]') -> 'Boolean':
        return Boolean(self._value >= other._value)

    @deref_decorator
    def __le__(self, other: 'NumericAtomic[T]') -> 'Boolean':
        return Boolean(self._value <= other._value)


@Atomic.register_atomic
class Real(NumericAtomic[float]):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.ALLOWED_ADDITIONAL_TYPES = (Integer, )

    def set_value(self, value: t.Union[int, float]) -> None:
        assert isinstance(value, (int, float))
        self._value = float(value)

    @deref_decorator
    def mutation(self, other: 'Real') -> 'mutations.Mutation':
        return mutations.RealMutation(self.value, other.value)

    def parse_literal(self, tokens: t.Sequence[lark.Token]):
        assert len(tokens) == 1
        self.set_value(float(tokens[0]))
        return self

    def stringify(self, scope: 'Compound',
                  name: t.Sequence[lark.Token]) -> str:
        return s.RealStringifier().stringify(self.value, scope, name)

    @deref_decorator
    def __truediv__(self, other):
        assert isinstance(other, Real)
        return Real(self._value / other._value)

    @property
    def _default(self) -> float:
        return 0.


@Atomic.register_atomic
class Integer(NumericAtomic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ALLOWED_ADDITIONAL_TYPES = (Real, )

    def set_value(self, value):
        assert isinstance(value, (int, float))
        self._value = int(value)

    @deref_decorator
    def mutation(self, other):
        return mutations.IntegerMutation(self.value, other.value)

    def parse_literal(self, tokens):
        assert len(tokens) == 1
        self.set_value(int(tokens[0]))
        return self

    def stringify(self, scope, name):
        return s.IntegerStringifier().stringify(self.value, scope, name)

    @deref_decorator
    def __truediv__(self, other):
        assert isinstance(other, Integer)
        return Integer(self._value // other._value)

    @deref_decorator
    def __mod__(self, other):
        assert isinstance(other, Integer)
        return Integer(self._value % other._value)

    @property
    def _default(self) -> int:
        return 0


@Atomic.register_atomic
class String(Atomic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ALLOWED_ADDITIONAL_TYPES = (
            Integer,
            Real,
        )

    def set_value(self, value):
        assert isinstance(value, str)
        self._value = value

    @deref_decorator
    def mutation(self, other):
        return mutations.StringMutation(self.value, other.value)

    @deref_decorator
    def __add__(self, other):
        return String(self._value + str(other._value))

    def parse_literal(self, tokens):
        assert len(tokens) == 1
        token = tokens[0]
        if not hasattr(token, "type"):
            self.set_value(str(token))
        elif token.type == "SHORT_STRING":
            self.set_value(token[1:-1])
        elif token.type == "LONG_STRING":
            self.set_value(token[3:-3])
        else:
            assert False
        return self

    def stringify(self, scope, name):
        return s.StringStringifier().stringify(self.value, scope, name)

    @property
    def _default(self):
        return ""


@Atomic.register_atomic
class Unit(Atomic):
    def __init__(self) -> None:
        super().__init__(None)
        del self._value

    def set(self, value):
        raise NotImplementedError("Can't modify Unit")

    def set_value(self, value):
        raise NotImplementedError("Can't modify Unit")

    def mutation(self, value):
        raise NotImplementedError("Can't modify Unit")

    @property
    def value(self):
        return None

    def parse_literal(self, tokens):
        assert len(tokens) == 1
        assert tokens[0] == "nil"
        return self

    def clone(self, **kwargs):
        # immutable, no point in copying...
        return self

    def stringify(self, scope, name):
        return s.UnitStringifier().stringify(self.value, scope, name)

    def parse(self, *args, **kwargs):
        yield from ()


@Atomic.register_atomic
class Boolean(Atomic):
    def set_value(self, value):
        assert isinstance(value, bool)
        self._value = value

    @deref_decorator
    def mutation(self, other):
        return mutations.BooleanMutation(self.value, other.value)

    def parse_literal(self, tokens):
        assert len(tokens) == 1
        assert str(tokens[0]).upper() in ("TRUE", "FALSE")
        self._value = ({"TRUE": True, "FALSE": False}[str(tokens[0]).upper()])
        return self

    def stringify(self, scope, name):
        return s.BooleanStringifier().stringify(self.value, scope, name)

    def __bool__(self):
        return self._value


@Atomic.register_atomic
class Flag(Boolean):
    ALLOWED_ADDITIONAL_TYPES = (Boolean, )

    def stringify(self, scope, name):
        return s.FlagStringifier().stringify(self.value, scope, name)

    @property
    def is_name_shown(self):
        return False


@Atomic.register_atomic
class Reference(Atomic):
    def __init__(self,
                 value,
                 scope=None,
                 cast=None,
                 do_not_collapse=False,
                 *args,
                 **kwargs):
        self._is_expression = False
        self._is_actual = False
        self._is_collapsed = False
        self._old_value = None
        self._cast = cast
        self._scope = scope
        self._do_not_collapse = False
        self._receiver = None
        super().__init__(value, *args, **kwargs)
        self._do_not_collapse |= do_not_collapse

    def set(self, other):
        if (not isinstance(other, Reference)
                and (self.type is None or isinstance(other, self.type))):
            return self.set_value(other)
        if not isinstance(other, self.__class__):
            raise TypeException("Type exception: {0} is not a {1}".format(
                type(other).__name__, self.__class__.__name__))
        clone = other.clone()
        self._value = clone._value
        self._scope = clone._scope
        self._is_expression = clone._is_expression
        self._is_actual = clone._is_actual
        self._do_not_collapse = clone._do_not_collapse

    def set_do_not_collapse(self):
        self._do_not_collapse = True

    def set_value(self, value):
        if (isinstance(value, str)
                or not isinstance(value, collections.abc.Collection)):
            return self._set_value_actual(value)
        if len(value) == 1 and isinstance(value[0], parser.Expression):
            return self._set_value_expression(value[0])
        return self._set_value_identifier(value)

    def _set_value_actual(self, value):
        assert self.type is not None or isinstance(value, Obj)
        was_expression = self._is_expression
        self._is_expression = False
        self._is_actual = True
        self._is_collapsed = False
        if isinstance(value, Reference):
            self.set(value)
        elif isinstance(value, Obj):
            self._value = value
        elif was_expression:
            self._value = self.type(value)
        else:
            self._value.set_value(value)
        self._do_not_collapse = False

    def _set_value_identifier(self, identifier):
        assert identifier[-1].type != "SCOPE_OPERATOR"
        assert len(identifier) > 0
        for token in identifier:
            assert isinstance(token, lark.Token)
        self._value = identifier
        self._is_actual = False
        self._is_collapsed = False
        self._is_expression = False
        self._do_not_collapse = False

    def _set_value_expression(self, expression):
        assert isinstance(expression, parser.Expression)
        self._value = expression
        self._is_actual = False
        self._is_expression = True
        self._is_collapsed = False
        self._do_not_collapse = True

    def mutation(self, other):
        #        assert isinstance(other, self.__class__)
        if issubclass(self.type, Atomic):
            new = other
            while isinstance(new.value, Atomic):
                new = new.value
            return self.value.mutation(new)
        return mutations.ReferenceMutation(self.value, other.value)

    def parse_literal(self, tokens):
        self._value = tokens
        return self

    def parse(self,
              tokens: t.Sequence[str],
              receiver: t.Optional[Obj] = None,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple[Obj, t.Any, t.Sequence[str]]]:
        return self.value.parse(tokens, receiver, options)

    def _dereference(self) -> t.Tuple[Obj, t.Optional[FindMetadata]]:
        if self._is_expression:
            assert self._do_not_collapse
            ret: t.Tuple[Obj, t.Optional[FindMetadata]] = (self._value.value,
                                                           None)
        elif self._is_actual:
            ret = (self._value, self._receiver)
        else:
            if self._scope is None:
                ret = (None, None)
            else:
                value, metadata = next(
                    self._scope.parse_full_match(
                        self._value, None,
                        FindOptions(search_environment=True)),
                    (None, None))  # first only
                assert value is not None, "lookup error"
                ret = value, metadata
        # Dereference recursively. Hopefully don't get stuck.
        while isinstance(ret[0], Reference):
            ret = ret[0]._dereference()
        if self._cast is not None and ret[0] is not None:
            if issubclass(self._cast, Reference):
                ret = self._cast(ret[0]), ret[1]
            else:
                ret = self._cast(ret[0].value), ret[1]
        # Cache the reference parse (this saves a ton of time)
        if not self._do_not_collapse and not self._is_collapsed:
            self._is_actual = True
            self._is_collapsed = True
            self._old_value = self._value
            self._value = ret[0]
            self._receiver = ret[1]
        return ret

    @property
    def value(self) -> Obj:
        return self._dereference()[0]

    @property
    def receiver(self) -> t.Optional[Obj]:
        ret = self._dereference()[1]
        if ret is not None:
            return ret.receiver
        return self._scope

    @property
    def is_name_shown(self):
        return self.value.is_name_shown

    def stringify(self, scope, name):
        return s.ReferenceStringifier().stringify(self.value, scope, name)

    @property
    def type(self) -> t.Optional[t.Type[Atomic]]:
        if (not hasattr(self, '_value') or self._value is None
                or self.value is None):
            return None
        return self.value.type

    def type_check(self, other):
        return self.value.type_check(other)

    def clone(self, scope=None, **kwargs):
        if scope is None:
            scope = self._scope
        if self._is_expression:
            return self.__class__((self._value.clone(scope=scope, **kwargs), ),
                                  scope,
                                  cast=self._cast,
                                  do_not_collapse=self._do_not_collapse)
        if self._is_collapsed:
            return self.__class__(self._old_value,
                                  scope,
                                  cast=self._cast,
                                  do_not_collapse=self._do_not_collapse)
        return self.__class__(self._value,
                              scope,
                              cast=self._cast,
                              do_not_collapse=self._do_not_collapse)

    def is_reference(self):
        return True

    @deref_decorator
    def __eq__(self, other):
        return self.value == other


class Named(Obj):
    def __init__(self, pattern, value, is_meta=False):
        self._pattern = pattern
        self._value = value
        self._is_meta = is_meta

    def find_actor(self,
                   tokens: t.Sequence[str],
                   receiver: t.Optional[Obj] = None,
                   options: FindOptions = FindOptions()
                   ) -> t.Iterator[t.Tuple[Obj, FindMetadata]]:
        unparsed, match, patterns = self._pattern.match(tokens)
        if receiver is None:
            receiver = self.value
        if not self.value.is_atomic():
            receiver = self.value
        if match and len(unparsed) == 0:
            yield self.value, FindMetadata(patterns=patterns,
                                           receiver=receiver,
                                           named=self)
        return None

    def find_character(self,
                       tokens: t.Sequence[str],
                       receiver: t.Optional[Obj] = None,
                       options: FindOptions = FindOptions()
                       ) -> t.Iterator[t.Tuple[Obj, FindMetadata]]:
        return self.find_actor(tokens, receiver)

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._value.type

    def type_check(self, other):
        return self._value.type_check(other)

    def begin(self):
        return self.value.begin()

    def parse(self,
              tokens: t.Sequence[lark.Token],
              receiver: t.Optional[Obj] = None,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple[Obj, t.Any, t.Sequence[lark.Token]]]:
        if receiver is None:
            receiver = self.value
        stop = len(tokens)
        for i, token in enumerate(tokens):
            if token.type == "SCOPE_OPERATOR":
                stop = i
                break
        else:
            return self.find_character(tokens, receiver, options)

        # For matches after a SCOPE_OPERATOR, search only inheritances and not
        # in the environment
        options = FindOptions(search_environment=False)
        for character, meta in self.find_character(tokens[:stop], receiver,
                                                   options):
            for ret in character.parse(tokens[stop:], meta.receiver, options):
                yield ret

    def clone(self, **kwargs):
        kwargs['first'] = True
        return self.__class__(self._pattern, self.value.clone(**kwargs),
                              self.is_meta)

    def stringify(self, scope, name):
        return self.value.stringify(scope, self._pattern._tokens)

    @property
    def is_name_shown(self):
        return self.value.is_name_shown

    @property
    def is_meta(self):
        return self._is_meta

    def __repr__(self):
        return f"Named('{' '.join(self._pattern._tokens)}', value={type(self.value).__name__}(...))"


class Compound(Obj):
    def __init__(self, command=None, actor=False, name=None, fqn=None):
        self.characters = []
        self.actors = []
        self._constructor = None
        self._history = []
        self._actor = actor
        if command is None:
            command = directives.Relax()
        self.command = command
        self._aggregate = False
        self._synthetic = False
        self._inherited_num = 0
        self.name = name
        self.fqn = fqn

    def _find(self,
              tokens: t.Tuple[lark.Token, ...],
              entries: t.Tuple[Obj, ...],
              method: str,
              receiver: Obj,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple[Obj, FindMetadata]]:
        for entry in entries:
            found = False
            if (not options.search_environment
                    and isinstance(entry, EnvironmentLink)):
                continue
            for ret, metadata in getattr(entry, method)(tokens, receiver,
                                                        options):
                found = True
                yield ret, metadata
            if found:
                return

    def find_actor(self,
                   tokens: t.Sequence[lark.Token],
                   receiver: t.Optional[Obj] = None,
                   options: FindOptions = FindOptions()
                   ) -> t.Iterator[t.Tuple['Obj', t.Any]]:
        if receiver is None:
            receiver = self
        return self._find(tuple(tokens), tuple(self.actors), 'find_actor',
                          receiver, options)

    def find_character(self,
                       tokens: t.Sequence[lark.Token],
                       receiver: t.Optional[Obj] = None,
                       options: FindOptions = FindOptions()
                       ) -> t.Iterator[t.Tuple['Obj', t.Any]]:
        if receiver is None:
            receiver = self
        return self._find(tuple(tokens), tuple(self.characters),
                          'find_character', receiver, options)

    def insert_character(self, obj: Obj, place: int) -> None:
        """Insert an `Obj` into the character list at a specified location.

        :param obj: The object to insert.
        :param place: The location to insert the character at.
        """
        self.characters.insert(place, obj)
        self._history.append((self.__class__.insert_character, obj, place))

    def insert_actor(self, obj: Obj, place: int):
        """Insert an `Obj` into the character and actor lists at a specified
        location.

        :param obj: The object to insert.
        :param place: The location to insert the character at.
        """
        self.actors.insert(place, obj)
        self.characters.insert(place, obj)
        self._history.append((self.__class__.insert_actor, obj, place))

    def append_character(self, obj):
        """Append an `Obj` to the end of the character list, but before
        inherited `Obj`s, if those exist.

        :param obj: The object to insert.
        """
        self.insert_character(
            obj,
            len(self.characters)
            if self._inherited_num == 0 else -self._inherited_num)

    def append_actor(self, obj):
        """Append an `Obj` to the end of the character and actor lists, but
        before inherited `Obj`s, if those exist.

        :param obj: The object to insert.
        """
        self.insert_actor(
            obj,
            len(self.actors)
            if self._inherited_num == 0 else -self._inherited_num)

    def append_actor_inherited(self, obj):
        """Append an `Obj` to the end of the character and actor lists, after
        any inherited `Obj`s, of those exist.

        :param obj: The object to insert.
        """

        self.insert_actor(obj, len(self.actors))
        self._inherited_num += 1

    def set_constructor(self, params):
        assert self._constructor is None
        self._constructor = tuple(params)

    def constructor_call(self,
                         arguments: t.Sequence[Obj],
                         token: lark.Token = None) -> None:
        if not self._constructor:
            assert len(arguments) == 0
            return
        if len(self._constructor) != len(arguments):
            raise CtorException(
                f"Constructor call at {token.line}:{token.column} has wrong"
                f" parameters (expected {len(self._constructor)},"
                f" got {len(arguments)})")
        assert len(self._constructor) == len(arguments)
        self._constructor = [(
            pattern,
            value,
        ) for (pattern, _), value in zip(self._constructor, arguments)]

    def constructor_param(self, name):
        if self._constructor is None:
            return None
        for pattern, value in self._constructor:
            if pattern.match(name)[1]:
                return value
        return None

    def begin(self):
        if self._actor or not isinstance(self.command, directives.Relax):
            return self.command
        return then_meanwhile_tree.Meanwhile(
            [self.command] + [obj.begin() for obj in self.actors])

    @property
    def value(self):
        return self

    @property
    def type(self):
        return self.__class__

    def type_check(self, other):
        # Type system goes here :)
        return isinstance(other, self.type)

    def parse(self,
              tokens: t.Sequence[lark.Token],
              receiver: t.Optional[Obj] = None,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple['Obj', t.Any, t.Sequence[str]]]:
        if receiver is None:
            receiver = self
        for i in range(0, len(tokens)):
            found = False
            for actor, metadata in self.find_actor(tokens[:i + 1], receiver,
                                                   options):
                found = True
                if (i + 1 < len(tokens)
                        and tokens[i + 1].type == "SCOPE_OPERATOR"):
                    found2 = False
                    subtokens = tokens[i + 2:]
                    if subtokens:
                        for subresult in actor.parse(subtokens,
                                                     metadata.receiver,
                                                     options):
                            found2 = True
                            yield subresult
                        if not found2:
                            found = False
                    else:
                        assert isinstance(actor, Reference)
                        yield actor.value, metadata, ()
                        # assert False, "not found?!"  # ??
                    # TODO: Throw exception if nothing found?
                else:
                    yield actor, metadata, tokens[i + 1:]
            if found:
                pass
                # break
        # TODO: Throw exception?
        return

    def inject(self, other: 'Compound') -> None:
        self.actors.extend(other.actors)
        self.characters.extend(other.characters)

    def clone(self,
              first=False,
              replacements: t.Dict[int, Obj] = None,
              inheritor: t.Optional['Compound'] = None,
              **kwargs) -> 'Compound':
        # code is immutable in Thapl, no need to copy command
        ret = self.__class__(self.command,
                             actor=self._actor,
                             name=f'{self.name}*')
        if replacements is None:
            replacements = {id(self): ret}
        else:
            if id(self) in replacements:
                return t.cast('Compound', replacements[id(self)])
            replacements[id(self)] = ret
        if first:
            scope = Compound()
            scope.append_actor(ret)
            if 'scope' in kwargs and kwargs['scope'] is not None:
                scope.append_actor(kwargs['scope'])
            kwargs['scope'] = scope
            kwargs['second'] = True
        for method, obj, place in self._history:
            method(
                ret,
                obj.clone(first=False,
                          replacements=replacements,
                          inheritor=inheritor,
                          **kwargs), place)
        ret._inherited_num = self._inherited_num
        ret._aggregate = self._aggregate
        ret._synthetic = self._synthetic
        ret._constructor = self._constructor
        return ret

    def enable_aggregate(self) -> None:
        """Make this `Compound` an aggregate `Compound`, that is mark it as
        containing nested `Obj`s.
        """
        self._aggregate = True

    def enable_synthetic(self) -> None:
        """Mark this `Compound` as a group alias that cannot be acted on
        directly.  Instead, any actions performed on it are passed on to its
        nested `Obj`s.
        """
        self._synthetic = True

    def stringify(self, scope: 'Compound',
                  name: t.Sequence[str]) -> t.Optional[str]:
        ret = [
            x for x in (s.CompoundStringifier().stringify(
                actor, scope, name if i == 0 and not self._synthetic else t.
                cast('Compound', actor).name)
                        for i, actor in enumerate(self.expand()))
            if x is not None
        ]
        if not ret:
            return None
        return "\n".join(ret)

    def all_has_properties(self, visited: t.Optional[t.FrozenSet[Obj]] = None
                           ) -> t.Iterator[Named]:
        """Find all 'has properties' contained in this `Compound`, following the
        rules for inheritance.
        """
        if visited is None:
            visited = frozenset()
        if self in visited:
            return  # StopIteration

        visited = visited | {self}
        for actor in self.actors:
            if isinstance(actor, Named):
                if (not actor.is_meta
                        and (not isinstance(actor.value, Compound)
                             or not actor.value._actor)):
                    yield actor
            elif isinstance(actor, InheritanceLink):
                for subactor in t.cast('Compound',
                                       actor.link).all_has_properties(visited):
                    yield subactor
            elif isinstance(actor, (EnvironmentLink, CanLink)):
                pass  # Don't look in the Environment
            else:
                import pdb
                pdb.set_trace()

    def grab_variables(self):
        """Find all variables contained in this `Compound` (probably a verb),
           and return them cloned. """
        ret = Compound()
        for actor in self.actors:
            if isinstance(actor, VariableLink):
                ret.append_actor(actor.clone())
        return ret

    def expand(self) -> t.Iterator[Obj]:
        if not self._synthetic:
            yield self
        if self._aggregate:
            visit: t.List[Obj] = []
            visit.extend(self.actors)
            while visit:
                current = visit.pop()
                if isinstance(current, Named) and len(
                        current._pattern._tokens
                ) == 2 and current._pattern._tokens[1] == "2nd":
                    import pdb
                    pdb.set_trace()
                if (isinstance(current, Named)
                        and isinstance(current.value, Compound)
                        and current.value._actor is True):
                    for ret in current.value.expand():
                        yield ret
                elif isinstance(current, (Compound, InheritanceLink)):
                    visit.extend(current.actors)

    def __repr__(self) -> str:
        return f"Compound(name={self.name},actors={self.actors},characters={self.characters})"


class Link(Obj):
    def __init__(self, linked: Obj):
        super().__init__()
        self.link = linked

    def parse(self,
              tokens: t.Sequence[str],
              receiver: t.Optional[Obj] = None,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple['Obj', FindMetadata, t.Sequence[str]]]:
        return self.link.parse(tokens, receiver, options)

    def begin(self, *args, **kwargs):
        return self.link.begin(*args, **kwargs)

    def find_actor(self,
                   tokens: t.Sequence[str],
                   receiver: t.Optional[Obj] = None,
                   options: FindOptions = FindOptions()
                   ) -> t.Iterator[t.Tuple['Obj', t.Any]]:
        return self.link.find_actor(tokens, receiver, options)

    def find_character(self,
                       tokens: t.Sequence[str],
                       receiver: t.Optional[Obj] = None,
                       options: FindOptions = FindOptions()
                       ) -> t.Iterator[t.Tuple['Obj', t.Any]]:
        return self.link.find_character(tokens, receiver, options)

    @property
    def value(self):
        return self.link.value

    @property
    def type(self):
        return self.link.type

    def type_check(self, *args, **kwargs):
        return self.link.type_check(*args, **kwargs)

    def clone(self, *args: t.Any, **kwargs: t.Any) -> 'Link':
        return type(self)(self.link.clone(*args, **kwargs))

    def stringify(self, *args, **kwargs):
        return self.link.stringify(*args, **kwargs)


class EnvironmentLink(Link):
    def parse(self,
              tokens: t.Sequence[str],
              receiver: t.Optional[Obj] = None,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple['Obj', t.Any, t.Sequence[str]]]:
        if id(self.link) in options.visited_envs:
            return iter(())
        return self.link.parse(
            tokens, self.link,
            options._replace(visited_envs=frozenset((id(self.link), ))
                             | options.visited_envs))

    def find_actor(self,
                   tokens: t.Sequence[str],
                   receiver: t.Optional[Obj] = None,
                   options: FindOptions = FindOptions()
                   ) -> t.Iterator[t.Tuple['Obj', t.Any]]:
        if id(self.link) in options.visited_envs:
            return iter(())
        return self.link.find_actor(
            tokens, self.link,
            options._replace(visited_envs=frozenset((id(self.link), ))
                             | options.visited_envs))

    def find_character(self,
                       tokens: t.Sequence[str],
                       receiver: t.Optional[Obj] = None,
                       options: FindOptions = FindOptions()
                       ) -> t.Iterator[t.Tuple['Obj', t.Any]]:
        if id(self.link) in options.visited_envs:
            return iter(())
        return self.link.find_character(
            tokens, self.link,
            options._replace(visited_envs=frozenset((id(self.link), ))
                             | options.visited_envs))

    def clone(self, second=False, inheritor=None, **kwargs):
        if not second or inheritor is None:
            return type(self)(self.link.clone(**kwargs))
        return type(self)(inheritor)


class InheritanceLink(Link):
    def expand(self, *args, **kwargs):
        return self.link.expand(*args, **kwargs)

    @property
    def actors(self):
        return self.link.actors


class ActorLink(Link):
    pass


class CanLink(Link):
    def stringify(self, *args, **kwargs):
        return None


class VariableLink(Link):
    def parse(self,
              tokens: t.Sequence[str],
              receiver: t.Optional[Obj] = None,
              options: FindOptions = FindOptions()
              ) -> t.Iterator[t.Tuple['Obj', t.Any, t.Sequence[str]]]:
        if options.search_variables:
            return super().parse(tokens, receiver, options)
        return ()
