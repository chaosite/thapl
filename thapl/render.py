#!/usr/bin/env python3
import math
import string

from thapl.log import CallbackVisitor
from thapl.parser import TopLevelParser

formatter = string.Formatter()


class Formatter:
    pass


class Renderer:
    header = r"""
\begin{frame}[fragile,t]
  \frametitle{}
  \begin{adjustbox}{}
    \begin{tikzpicture}"""
    footer = r"""    \end{tikzpicture}
  \end{adjustbox}
\end{frame}"""

    def __init__(self, log, obj):
        self._events = []
        self._max_time = 0
        self._log = log
        self._obj = obj

        self._log_by_frame = None
        self._parser = TopLevelParser()
        self.discretize()

    def render(self):
        self._reset_obj()

        active_logs = {}
        ended_logs = []
        rendered = []

        def event_start(event_time, event_log):
            active_logs[id(event_log)] = event_log

        def event_key(event_time, event_log):
            for event_log in ended_logs:
                event_log.field_update_to_time(event_log.stop)
            ended_logs.clear()

            for event_log in active_logs.values():
                event_log.field_update_to_time(event_time)

            rendered.append(self.header)
            rendered.append(self._render_obj())
            rendered.append(self.footer)

        def event_stop(event_time, event_log):
            del active_logs[id(event_log)]
            ended_logs.append(event_log)

        switch = {"start": event_start, "key": event_key, "stop": event_stop}

        for time, event, log in self._events:
            switch[event](time, log)

        rendered.append("")
        return "\n".join(rendered)

    def _add_single_log(self, log):
        self._events.append((log.start, "start", log))
        self._events.append((log.stop, "stop", log))

        self._max_time = max(self._max_time, log.stop)

    def discretize(self):
        self._events = []
        self._max_time = 0
        CallbackVisitor(self._add_single_log).visit(self._log)
        self._events.extend((i, "key", None)
                            for i in range(int(math.ceil(1 + self._max_time))))
        self._events.sort(key=lambda x: x[0])

    def _reset_obj(self):
        slot_cache = set()
        for frame in self._events:
            if frame[1] == "start":
                log = frame[2]
                if id(log.field) in slot_cache:
                    continue
                slot_cache.add(id(log.field))
                log.field_reset()

    def _render_obj(self):
        ret = []
        for entry in self._obj.value.actors[::-1]:
            maybe = self._render_actor(entry)
            if maybe:
                ret.append(maybe)
        return "\n".join(ret)

    def _render_actor(self, named):
        return named.stringify(self._obj, None)
