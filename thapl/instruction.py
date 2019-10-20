import numpy as np
import typing as t

from thapl import then_meanwhile_tree, context
from thapl.log import Log
from thapl.pattern import Pattern
from thapl.spring import Spring

ϵ = np.nextafter(0, 1)

if t.TYPE_CHECKING:
    from thapl import directives, mutations


class Instruction(then_meanwhile_tree.Atomic):
    """An abstract class representing a single atomic instruction.  Several
    instructions are composed together using the general ThenMeanwhileTree
    structure.  A program thus makes a tree of this sort, where each of the
    leaves has an instance of this class (or any of its subclasses).

    :param directive: The directive that created this instruction, for
    debugging purposes.

    """
    def __init__(self,
                 directive: t.Optional['directives.AtomicDirective'] = None
                 ) -> None:
        self._directive = directive


class Relax(Instruction):
    """ Represents a no-operation instruction """
    def execute(self, obj: 'context.Obj') -> Log:
        return Log(None, Spring(k=ϵ, natural=0, minimum=0), None)


class Change(Instruction):
    """ Instruction representation of `box_a set x to 20`. """
    def __init__(self,
                 subject: 'context.Atomic',
                 value: 'context.Obj',
                 scope: 'context.Compound',
                 length: t.Optional['context.Atomic'],
                 k: t.Optional['context.Atomic'],
                 directive: t.Optional['directives.AtomicDirective'] = None
                 ) -> None:
        super().__init__(directive)
        self._subject = subject
        self._value = value
        self._scope = scope
        self._length = length
        self._k = k

    def execute(self, obj: 'context.Obj') -> Log:
        value = self._scopify(self._value)
        mutation = self._subject.mutation(value)
        self._subject.set(value)
        return Log(mutation, self._spring(mutation), self._subject)

    def _scopify(self, value: 'context.Atomic') -> 'context.Atomic':
        if isinstance(value, context.Reference):
            return value.clone(scope=self._scope)
        return value

    def _spring(self, mutation: 'mutations.Mutation') -> Spring:
        k, natural = 1, abs(mutation.delta)
        if self._length:
            natural = self._scopify(self._length)
            while isinstance(natural, context.Atomic):
                natural = natural.value
        if self._k:
            k = self._scopify(self._k)
            while isinstance(k, context.Atomic):
                k = k.value
        return Spring(k=k, natural=natural)

    def __repr__(self) -> str:
        return "{}({}, {}, {})".format(
            type(self).__name__, self._subject, self._value, self._scope)


class Set(Change):
    """ Set something that doesn't show up on screen. """
    def _spring(self, _: 'mutations.Mutation') -> Spring:
        return Spring(k=ϵ, natural=0, minimum=0)

    def execute(self, obj: t.Optional['Context.obj']):
        super().execute(obj)
        return

    @classmethod
    def from_change(cls, change: Change) -> 'Set':
        return cls(change._subject,
                   change._value,
                   change._scope,
                   change._length,
                   change._k,
                   directive=change._directive)


class Call(Instruction):
    """ A call to a sub section """
    def __init__(self,
                 call_directive: then_meanwhile_tree.Compound,
                 call_obj: 'context.Compound',
                 directive: t.Optional['directives.AtomicDirective'] = None
                 ) -> None:
        super().__init__(directive)
        self._call_directive = call_directive
        self._call_obj = call_obj

    def execute(self, obj: 'context.Obj') -> Log:
        # TODO: Interpret here or before?
        return t.cast(
            Log,
            self._call_directive.reduce(
                then_meanwhile_tree.InterpretTransformer(
                    self._call_obj)).reduce(
                        then_meanwhile_tree.ExecuteTransformer(
                            self._call_obj)))

    def __repr__(self) -> str:
        return "Call(...)"


class Invocation(Instruction):
    """Represents an executed atomic directive containing precisely one subject,
    one verb, and a possibly empty list of modifiers. For example, the Thapl
    input `box_a moves 2 spaces left.` generates an instance of this class, in
    which the subject is box_a, the verb is moves, and two modifiers: 2 spaces
    and left.

    Note that the directive `king and queen bow` will generate two
    instances of this class, connected with a meanwhile clause.

    """
    def __init__(self, subject: 'context.Compound', verb: 'context.Compound',
                 modifiers: t.Sequence[
                     t.Tuple['context.Obj', t.Dict[str, 'context.Atomic']]],
                 directive: t.Optional['directives.AtomicDirective']) -> None:
        super().__init__(directive)
        self._subject = subject
        self._verb = verb
        self._modifiers = modifiers

    def execute(self, obj: 'context.Obj') -> Log:
        # TODO: Is interpreting here again fine? Or should it be done ahead of
        # time?
        new_obj = self._create_obj(obj, self._verb)
        return then_meanwhile_tree.Then([
            self._interpret_and_execute(
                modifier.begin(),
                self._create_obj_for_modifier(new_obj, metadata))
            for modifier, metadata in self._modifiers
            if not isinstance(modifier, context.Unit)
        ] + [self._interpret_and_execute(self._verb.begin(), new_obj)])

    def _interpret_and_execute(self, directive: 'directives.AtomicDirective',
                               obj: 'context.Obj') -> Log:
        return directive.reduce(
            then_meanwhile_tree.InterpretTransformer(obj)).reduce(
                then_meanwhile_tree.ExecuteTransformer(obj))

    def _create_obj(self, orig_obj: 'context.Obj',
                    verb: 'context.Compound') -> 'context.Obj':
        obj = context.Compound()
        for value, metadata in self._modifiers:
            # If this is a "global" modifier, just process it.
            if isinstance(value, context.Unit):
                for name, literal in metadata.items():
                    obj.append_actor(context.Named(Pattern([name]), literal))
        obj.append_actor(context.Named(Pattern(["self"]), self._subject))
        obj.append_actor_inherited(self._subject.value)
        obj.append_actor_inherited(self._verb.grab_variables())
        obj.append_actor_inherited(orig_obj.value)
        return obj

    def _create_obj_for_modifier(self, new_obj: 'context.Obj',
                                 metadata: t.Dict[str, 'context.Atomic']
                                 ) -> 'context.Compound':
        obj = context.Compound()
        for name, literal in metadata.items():
            obj.append_actor(context.Named(Pattern[name]), literal)
        obj.append_actor_inherited(new_obj.value)
        return obj
