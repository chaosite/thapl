import math
import warnings
import typing as t

import numpy as np

# This can happen, but it's actually fine since it's defined in the floating
# point rules and does the right thing -- giving me an inf, which is fine.
warnings.filterwarnings("ignore", "overflow encountered in double_scalars")


class Interval:
    "Represents a (time) interval"

    def __init__(self, start: float, stop: float) -> None:
        assert start <= stop
        self._start = start
        self._stop = stop

    @property
    def start(self) -> float:
        return self._start

    @property
    def stop(self) -> float:
        return self._stop

    @property
    def duration(self) -> float:
        return self.stop - self.start

    def within(self, t: float) -> bool:
        return t <= self.stop and t >= self.start

    def __repr__(self) -> str:
        return "Interval({}, {})".format(self.start, self.stop)


class Spring:
    def __init__(self, natural: float, k: float, minimum: float = 1.) -> None:
        if natural < minimum:
            natural = minimum
        self._natural = natural
        self._minimum = minimum
        self._k = k
        assert k != 0
        assert not np.isnan(natural)
        assert not np.isnan(minimum)
        # Detached springs start at 0 and dangle for their natural length.
        self._interval = Interval(0, natural)

    @property
    def natural(self) -> float:
        return self._natural

    @property
    def minimum(self) -> float:
        return self._minimum

    @property
    def k(self) -> float:
        return self._k

    @property
    def interval(self) -> Interval:
        return self._interval

    def stretch(self, interval: Interval) -> None:
        self._interval = interval
        self._sift_down()

    def solve(self) -> None:
        self._sift_down()

    def _sift_down(self) -> None:
        pass

    def __repr__(self) -> str:
        return "{}(k = {}, natural = {}, minimum = {}, interval = {})".format(
            type(self).__name__, self._k, self._natural, self._minimum,
            self._interval)


class EquivalentSpring(Spring):
    def __init__(self, natural: float, k: float, minimum: float,
                 subsprings: t.Sequence[Spring]) -> None:
        super().__init__(natural, k, minimum)
        assert len(subsprings) >= 2
        self._subsprings = subsprings

    @property
    def subsprings(self) -> t.Sequence[Spring]:
        return self._subsprings


class SerialSpring(EquivalentSpring):
    def __init__(self, subsprings: t.Sequence[Spring]) -> None:
        self._cs = [1 / s.k if s.k != 0 else math.inf for s in subsprings]
        k = 1 / sum(self._cs)
        if k == 0.:
            k = np.nextafter(0, 1)  # k can't be 0, make it epsilon
        super().__init__(sum(s.natural for s in subsprings), k,
                         sum(s.minimum for s in subsprings), subsprings)

    def _sift_down(self) -> None:
        def partition(ki: float, naturali: float) -> float:
            if self.k == math.inf:
                assert not np.isnan(
                    self.interval.duration / len(self.subsprings))
                return self.interval.duration / len(self.subsprings)
            return max(0,
                       (naturali +
                        (self.interval.duration - self.natural) * self.k / ki))

        assert self.interval.duration >= self.minimum or np.isclose(
            self.interval.duration, self.minimum)
        start = self.interval.start
        for s in self.subsprings[:-1]:
            interval = Interval(start, start + partition(s.k, s.natural))
            # if interval.duration < s.minimum:
            #     interval._stop = start + s.minimum
            s.stretch(interval)
            start = interval.stop

        interval = Interval(min(start, self.interval.stop), self.interval.stop)
        self.subsprings[-1].stretch(interval)
        assert (
            abs(self.subsprings[-1].interval.duration -
                partition(self.subsprings[-1].k, self.subsprings[-1].natural))
            < 3.0)


class ParallelSpring(EquivalentSpring):
    def __init__(self, subsprings: t.Sequence[Spring]) -> None:
        k_eq = sum(s.k for s in subsprings)
        l_eq = sum(s.interval.duration * s.k for s in subsprings) / k_eq
        super().__init__(l_eq, k_eq, max(s.minimum for s in subsprings),
                         subsprings)

    def _sift_down(self) -> None:
        for s in self.subsprings:
            s.stretch(self.interval)
