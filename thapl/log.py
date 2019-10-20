from thapl.then_meanwhile_tree import Atomic, BottomupVisitor

import math


class Log(Atomic):
    """
    A log contains a record of an instruction after it was executed. Log entries
    are stored in a ThenMeanwhileTree.
    """

    def __init__(self, mutation, spring, field, *args, **kw):
        super().__init__(*args, **kw)

        self.mutation = mutation
        self.spring = spring
        self.field = field

    def __repr__(self):
        return "Log({}, {})".format(self.mutation, self.spring)

    def field_update_to_time(self, t):
        if self.mutation is None:
            return
        if t == int(self.spring.interval.stop) + 1:
            t = self.spring.interval.stop
        value = self.mutation.x(t)
        assert not isinstance(value, float) or not math.isnan(value)
        if value is not None:
            self.field.set_value(value)

    def field_reset(self):
        self.field_update_to_time(self.spring.interval.start)

    def discrete_frames(self):
        return range(
            int(self.spring.interval.start),
            int(self.spring.interval.stop + 1))

    @property
    def start(self):
        return self.spring.interval.start

    @property
    def stop(self):
        return self.spring.interval.stop


class CallbackVisitor(BottomupVisitor):
    def __init__(self, callback):
        self._callback = callback

    def atomic(self, log):
        self._callback(log)

    def meanwhile(self, t):
        pass

    def then(self, t):
        pass
