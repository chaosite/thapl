"""
Regex compiler that compiles a thapl pattern and runs it on a token list input.
"""
import typing as t
from abc import ABC, abstractmethod

from lark import Token


class NFA:
    def __init__(self, pattern: t.Sequence[t.Union[Token, str]]) -> None:
        self._pattern = pattern
        self.compile()

    def _connect_previous(self, stack: t.List['Fragment'],
                          in_tag: int = -1) -> None:
        # No parens version
        if len(stack) > 1:
            e2 = stack.pop()
            e1 = stack.pop()
            for patch in e1.out:
                patch((e2.start, in_tag))
            f = Fragment(e1.start)
            f.out.extend(e2.out)
            stack.append(f)

    def _add_match_any(self, stack: t.List['Fragment'],
                       in_tag: int = -1) -> None:
        s = MatchAny(out=None, tag=in_tag)
        f = Fragment(s)
        f.out.append(s.patch)
        stack.append(f)

    def _add_kleene_star(self, stack: t.List['Fragment'],
                         out_tag: int = -1) -> None:
        e = stack.pop()
        s = Split(out=(e.start, -1), out1=None)
        for patch in e.out:
            patch((s, -1))
        f = Fragment(e.start)
        f.out.append(lambda t: s.patch1((t[0], out_tag)))
        stack.append(f)

    def _add_match_literal(self, stack: t.List['Fragment'],
                           literal: t.Sequence[str]) -> None:
        s = MatchLiteral(match=literal, out=None)
        f = Fragment(s)
        f.out.append(s.patch)
        stack.append(f)

    def compile(self) -> None:
        stack: t.List['Fragment'] = []
        counter = 0
        d: t.Dict[t.Any, t.Tuple[int, int]] = {}
        for token in self._pattern:
            if token.type == "VARIABLE":
                self._add_match_any(stack, in_tag=counter)
                self._add_kleene_star(stack, out_tag=counter + 1)
                self._connect_previous(stack, in_tag=counter)
                d[token] = (
                    counter,
                    counter + 1,
                )
                counter += 2
            else:  # TODO: add other operators
                self._add_match_literal(stack, token)
                self._connect_previous(stack)
        s = End()
        e = stack.pop()
        for patch in e.out:
            patch((s, -1))
        assert len(stack) == 0
        self._variable_to_tags: t.Dict[t.Any, t.Tuple[int, int]] = d
        self._start: 'State' = e.start

    def match(self, tokens: t.Sequence[t.Union[str, Token]]
              ) -> t.Optional[t.Dict[t.Any, t.Sequence[t.Any]]]:
        self.clist = StateList.start(self._start)
        self.nlist = StateList.start()
        tokens = list(tokens)
        for i, token in enumerate(tokens):
            self.step(token, i)
            self.clist, self.nlist = self.nlist, self.clist
        for state, vmap in self.clist:
            if isinstance(state, End):
                return self.convert_map(tokens, vmap)
        return None

    def convert_map(self, tokens: t.Sequence[t.Union[str, Token]],
                    vmap: t.Dict[int, int]
                    ) -> t.Dict[t.Any, t.Sequence[t.Any]]:
        ret = {}
        for variable in self._variable_to_tags:
            start_tag, stop_tag = self._variable_to_tags[variable]
            ret[variable] = tokens[vmap[start_tag] + 1:vmap[stop_tag] + 1]
        return ret

    def step(self, token: t.Union[str, Token], i: int) -> None:
        self.nlist = StateList.start()
        for state, vmap in self.clist:
            if state.match(token):
                assert state.out is not None
                self.nlist.add(state.out, vmap, i)


class StateList:
    listid = 0

    def __init__(self) -> None:
        self.states: t.List[t.Tuple['State', t.Dict[int, int]]] = []

    def add(self, _s: t.Tuple['State', int], vmap: t.Dict[int, int],
            value: t.Any) -> None:
        s, tag = _s
        if s is None or s.lastid == self.listid:
            return
        nvmap = {}
        nvmap.update(vmap)
        if tag != -1:
            nvmap[tag] = value
        s.lastid = self.listid
        if isinstance(s, Split):
            assert s.out is not None
            assert s.out1 is not None
            self.add(s.out, nvmap, value)
            self.add(s.out1, nvmap, value)
        else:
            self.states.append((s, nvmap))

    @classmethod
    def start(cls, s: t.Optional['State'] = None) -> 'StateList':
        cls.listid += 1
        ret = cls()
        if s is not None:
            nvmap = {}
            if s.tag != -1:
                nvmap[s.tag] = -1
            ret.add((s, -1), nvmap, None)
        return ret

    def __iter__(self) -> t.Iterator[t.Tuple['State', t.Dict[int, int]]]:
        return iter(self.states)


class State(ABC):
    def __init__(self, tag: int = -1) -> None:
        super().__init__()
        self.lastid = -1
        self.out: t.Optional[t.Tuple['State', int]]
        self.start: 'State'
        self.tag = tag

    @abstractmethod
    def match(self, token: t.Union[str, Token]) -> t.Optional[bool]:
        pass


class Split(State):
    def __init__(self, out: t.Optional[t.Tuple['State', int]],
                 out1: t.Optional[t.Tuple['State', int]]) -> None:
        super().__init__()
        self.out = out
        self.out1 = out1

    def patch(self, new: t.Tuple['State', int]) -> None:
        self.out = new

    def patch1(self, new: t.Tuple['State', int]) -> None:
        self.out1 = new

    def match(self, token: t.Union[str, Token]) -> None:
        assert False


class Single(State):
    def __init__(self, out: t.Optional[t.Tuple[State, int]],
                 tag: int = -1) -> None:
        super().__init__(tag)
        self.out = out

    def patch(self, new: t.Tuple[State, int]) -> None:
        self.out = new


class MatchLiteral(Single):
    def __init__(self,
                 match: t.Union[str, Token],
                 out: t.Optional[t.Tuple[State, int]],
                 tag: int = -1) -> None:
        super().__init__(out, tag)
        self._match = match

    def match(self, token: t.Union[str, Token]) -> bool:
        return self._match == token


class MatchAny(Single):
    def __init__(self, out: t.Optional[t.Tuple[State, int]],
                 tag: int = -1) -> None:
        super().__init__(out, tag)

    def match(self, token: t.Union[str, Token]) -> bool:
        return True


class End(State):
    def __init__(self, tag: int = -1) -> None:
        super().__init__(tag)

    def match(self, token: t.Union[str, Token]) -> bool:
        return False


class Fragment:
    def __init__(self, start: State) -> None:
        self.start: State = start
        self.out: t.List[t.Callable[[t.Tuple[State, int]], None]] = []
