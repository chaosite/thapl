from itertools import filterfalse

from lark import Lark, Transformer, v_args
from lark.exceptions import VisitError

from thapl import parser

EXPR_PARSER = Lark(
    r"""
start: ANYTHING* (top_inner ANYTHING*)*
top_inner: OPEN inner CLOSE
inner: ANYTHING* (OPEN inner CLOSE ANYTHING*)*

ANYTHING.1: /./| (CR? LF)
OPEN.10: "(("
CLOSE.10: "))"

COMMENT: "%(" /(.|\n|\r)*?/ ")%"
%ignore COMMENT

%import common.CR
%import common.LF
""",
    parser="lalr",
    start="start")
TOKEN_PARSER = None


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen.add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen.add(k)
                yield element


class StringParser(Transformer):
    def parse(self, text, scope, name):
        global TOKEN_PARSER
        if TOKEN_PARSER is None:
            TOKEN_PARSER = parser.open_parser(start="tokens")
        self._scope = scope
        self._name = name
        self._used_keys = {("render", ), ("visible", ), ("rendering", ),
                           ("i", ), ("j", ), ("k", ), ("l", )}
        return self.transform(EXPR_PARSER.parse(text))

    def _dereference(self, parsed):
        if parsed.is_reference():
            name = parsed._value
            if len(name) >= 3 and str(name[-3]) == "_":
                assert name[-2].type == "SCOPE_OPERATOR"
                if len(name) == 3 and str(name[-1]) == "keys":
                    return ("keys", )  # Mark it
                if len(name) == 3 and str(name[-1]) == "contents":
                    return ("contents", )
                assert str(name[-1]) == "name"
                assert len(name) == 3, "TODO: Restore functionality"
                return " ".join(self._name)
            else:
                self._used_keys.add(tuple(str(token) for token in name))
                # also add the first part in case we only used sub parts
                key = []
                for token in name:
                    if token.type == "SCOPE_OPERATOR":
                        self._used_keys.add(tuple(key))
                        break
                    key.append(str(token))
                return parsed.value.stringify(parsed.receiver, name)
        return parsed.stringify(self._scope, None)

    def _key_replace(self, result):
        if not isinstance(result, tuple):
            return result

        if result[0] == "keys":

            def stringify_key(actor):
                stringified = actor.stringify(self._scope, self._name)
                if stringified is None:
                    return None
                if not actor.is_name_shown:
                    return stringified
                return "{}={}".format(" ".join(actor._pattern._tokens),
                                      stringified)

            ret = (stringify_key(actor) for actor in unique_everseen(
                self._scope.all_has_properties(),
                key=lambda a: tuple(a._pattern._tokens)) if (tuple(
                    map(str, actor._pattern._tokens)) not in self._used_keys))
            return ",".join(filter(None, ret))
        if result[0] == "contents":
            ret = (actor.stringify(self._scope, self._name)
                   for actor in self._scope.all_has_properties() if tuple(
                       str(token) for token in actor._pattern._tokens) not in
                   self._used_keys)
            return "\n".join(filter(None, ret))

    # Rule transforms

    @v_args(inline=True)
    def top_inner(self, _open, unparsed, _close):
        tokens = TOKEN_PARSER.parse(unparsed).children
        parsed = parser.parse_literal(tokens, self._scope)
        return self._dereference(parsed)

    def inner(self, items):
        return "".join(items)

    def start(self, items):
        items = [self._key_replace(item) for item in items if item is not None]
        return "".join(items)


class StringExpression:
    _PARSER = None

    def __init__(self, fmt):
        self.fmt = fmt
        if self._PARSER is None:
            self._PARSER = StringParser()

    def evaluate(self, scope, name):
        try:
            return self._PARSER.parse(self.fmt, scope, name)
        except VisitError as e:
            raise e.orig_exc


class AtomicStringifier:
    def stringify(self, value, scope, name):
        return self.stringify_value(value, scope, name)

    def stringify_value(self, value, scope, name):
        return "{}".format(value)

    def _stringify_name(self, name):
        return " ".join(name)


class RealStringifier(AtomicStringifier):
    def stringify_value(self, value, scope, name):
        return "{:.6f}".format(value)


class IntegerStringifier(AtomicStringifier):
    pass


class StringStringifier(AtomicStringifier):
    def stringify_value(self, value, scope, name):
        if "((" in value:
            value = StringExpression(value).evaluate(scope, name)
        if "," in value:
            return "{{{}}}".format(value)
        return value


class UnitStringifier(AtomicStringifier):
    def stringify_value(self, value):
        return None


class BooleanStringifier(AtomicStringifier):
    def stringify_value(self, value, scope, name):
        return "true" if value else "false"


class ReferenceStringifier(AtomicStringifier):
    def stringify(self, value, scope, name):
        # return None
        return value.stringify(scope, name)


class FlagStringifier(AtomicStringifier):
    def stringify(self, value, scope, name):
        if value:
            return self._stringify_name(name)
        return None


class CompoundStringifier:
    def stringify(self, value, scope, name):
        # Importing here to prevent cyclical dependencies
        from thapl.context import FindOptions
        render_expr = None
        for maybe, _ in value.find_actor(
                ["render"], FindOptions(search_environment=False)):
            render_expr = maybe
            break
        if render_expr is None:
            return None
        render = render_expr.value
        if not isinstance(render, str):
            render = render.value
        return StringExpression(render).evaluate(value, name)
