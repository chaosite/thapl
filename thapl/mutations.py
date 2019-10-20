#!/usr/bin/env python3
"""
Mutations are objects that represent the ability to progressively modify a
context at some later date.
"""

import math
import numpy as np
import re
from itertools import chain

import sympy

MUTATION_COUNTER = 0


class MutationImpl:
    def __init__(self, mutation):
        self.mutation = mutation

    def create_equations(self):
        glob_sym = lambda x: sympy.Symbol("mut{}_{}".format(
            self.mutation._id, x))

        def create_equations(name,
                             prev_x=None,
                             prev_v=None,
                             prev_a=None,
                             prev_j=None,
                             prev_t=None,
                             end_x=None):
            ret = []
            sym = lambda x: sympy.Symbol("mut{}_{}_{}".format(
                self.mutation._id, name, x))
            varz = {}
            t = varz['t'] = sym("t")
            if prev_t is not None:
                start = varz['start'] = prev_t
            else:
                start = varz['start'] = sym("start")
            if prev_j is not None:
                j0 = varz['j0'] = prev_j
            else:
                j0 = varz['j0'] = sym("j0")
            a = varz['a'] = sym("a")
            if prev_a is not None:
                a0 = varz['a0'] = prev_a
            else:
                a0 = varz['a0'] = sym("a0")
            v = varz['v'] = sym("v")
            if prev_v is not None:
                v0 = varz['v0'] = prev_v
            else:
                v0 = varz['v0'] = sym("v0")
            x = varz['x'] = sym("x")
            if prev_x is not None:
                x0 = varz['x0'] = prev_x
            else:
                x0 = varz['x0'] = sym("x0")
            dt = t - start
            ret.append(j0 * dt + a0 - a)
            ret.append(j0 * dt**2 / 2 + dt * (a0) + v0 - v)
            ret.append(j0 * dt**3 / 6 + (a0) * dt**2 / 2 + (v0) * dt + x0 - x)
            return (varz, ret)

        ret = []
        total_dx = self.mutation.delta

        init_vars, init_eqs = create_equations(
            'init', prev_x=self.mutation.old, prev_v=0)
        cruise_vars, cruise_eqs = create_equations(
            'cruise',
            prev_x=self.mutation.old + total_dx * sympy.S("1") / 3,
            prev_a=0,
            prev_j=0)
        fini_vars, fini_eqs = create_equations(
            'fini', prev_x=self.mutation.old + total_dx * sympy.S("2") / 3)
        cruise_eqs = [
            eq.subs(((cruise_vars['j0'], 0), (cruise_vars['a0'], 0),
                     (cruise_vars['a'], 0))) for eq in cruise_eqs
        ]
        ret.extend(chain(init_eqs, cruise_eqs, fini_eqs))

        total_dt = fini_vars['t'] - init_vars['start']

        # Boundary conditions for phase transitions
        for (first_vars,
             first_eqs), (second_vars,
                          second_eqs) in (((init_vars, init_eqs),
                                           (cruise_vars, cruise_eqs)),
                                          ((cruise_vars, cruise_eqs),
                                           (fini_vars, fini_eqs))):
            for first_eq, second_eq, var_to_sub in zip(
                    first_eqs, second_eqs,
                    [(first_vars[v], second_vars[v]) for v in ['a', 'v', 'x']]):
                if cruise_vars['a'] in var_to_sub:
                    continue
                eq = first_eq.subs(var_to_sub[0],
                                   second_eq.subs(var_to_sub[1], 0))
                eq = eq.subs(((first_vars['x'],
                               second_vars['x0']), (first_vars['v'],
                                                    second_vars['v0']),
                              (first_vars['a'],
                               second_vars['a0']), (first_vars['t'],
                                                    second_vars['start']),
                              (second_vars['x'],
                               second_vars['x0']), (second_vars['v'],
                                                    second_vars['v0']),
                              (second_vars['a'],
                               second_vars['a0']), (second_vars['t'],
                                                    second_vars['start'])))
                ret.append(eq)

        self._eqs = ret
        self._vars = {
            'init': init_vars,
            'cruise': cruise_vars,
            'fini': fini_vars
        }
        self._phase_eqs = {
            'init': init_eqs,
            'cruise': cruise_eqs,
            'fini': fini_eqs
        }

    def supply(self):
        if self.mutation.duration == 0 or self.mutation.is_noop():
            return

        # Add equations for the ending boundary conditions
        # Note: We skip the first equation because... Well I don't know why.
        #       It confuses the solver somehow.
        for eq in self._phase_eqs['fini'][1:]:
            self._eqs.append(
                eq.subs((
                    (self._vars['fini']['a'], 0),
                    (self._vars['fini']['v'], 0),
                    (self._vars['fini']['x'], self.mutation.new),
                    (self._vars['fini']['t'], self.mutation.interval.stop),
                )))

        # Add the missing equation for defining phase length in time
        self._time_eqs = []
        self._time_eqs.append(
            sympy.Eq(self._vars['init']['start'],
                     self.mutation.interval.start))
        self._time_eqs.append(
            sympy.Eq(
                self.mutation.interval.stop - self._vars['fini']['start'],
                self._vars['fini']['start'] - self._vars['cruise']['start']))
        self._time_eqs.append(
            sympy.Eq(
                self._vars['cruise']['start'] - self._vars['init']['start'],
                self._vars['fini']['start'] - self._vars['cruise']['start']))
        # We know init_start, this acts as the starting boundary condition.

        self.solve()

    def solve(self):
        # the symbols that we are going to solve for. Everything except for 't'
        # and 'x', which are our parameters, and "start", which we solve for
        # seperately.
        symbols = [
            self._vars[p][v] for p in self._vars for v in self._vars[p]
            if isinstance(self._vars[p][v], sympy.Symbol) and v not in (
                "x", "t", "start")
        ]

        # Solve for start seperately. This isn't strictly needed but it's faster.
        time_symbols = [self._vars[p]['start'] for p in self._vars]
        time_solution = tuple(sympy.linsolve(self._time_eqs, time_symbols))[0]
        time_substitutions = dict(zip(time_symbols, time_solution))
        self._eqs = [eq.subs(time_substitutions.items()) for eq in self._eqs]
        # filter out always true equations
        self._eqs = [eq for eq in self._eqs if eq != 0]

        # HACK: linsolve is finicky and doesn't like redundant equations, even
        #       though they don't actually make the equation set inconsistent.
        #       nonlinsolve can handle them, but it's much slower...
        bad_eqs = [self._eqs[9], self._eqs[11], self._eqs[13]]
        self._eqs = (self._eqs[:9] + self._eqs[10:11] + self._eqs[12:13] +
                     self._eqs[14:])

        solutions = sympy.linsolve(self._eqs, *symbols)
        # 'd' is a short alias for the parameter dictionary.
        self._params = d = {**time_substitutions}
        # Assuming that all the solutions are the same (they are)
        for sol, sym in zip(tuple(solutions)[0], symbols):
            if sym.name[-1] not in ("0", ) or sym in sol.free_symbols:
                continue
            d[sym] = sol

        for symbol in d:
            d[symbol] = d[symbol].subs(
                ((self._vars['init']['t'], d[self._vars["cruise"]["start"]]),
                 (self._vars['init']['x'], self._vars["cruise"]["x0"]),
                 (self._vars['cruise']['x'], self._vars["fini"]["x0"]),
                 (self._vars['cruise']['t'], d[self._vars["fini"]["start"]]),
                 (self._vars['fini']['x'], self.mutation.new),
                 (self._vars['fini']['t'],
                  self.mutation.interval.stop)), ).simplify()

        x_equations = [l[-1] for l in self._phase_eqs.values()]
        x_equations = [eq.subs(d.items()).simplify() for eq in x_equations]

        # Get the motion equations as expressions of x (so we just plug t in)
        x_solutions = sympy.linsolve(
            x_equations, [self._vars[phase]['x'] for phase in self._vars])
        x_expressions = tuple(x_solutions)[
            0]  # grab first (and only) solution.
        # order them nicely in a mapping, phase -> x expression
        x_expressions = {
            re.search("_(.*)_",
                      tuple(expr.free_symbols)[0].name).group(1): expr
            for expr in x_expressions
        }
        phase_order = (
            "init",
            "cruise",
            "fini",
        )
        phase_boundaries = {
            "init": (self.mutation.interval.start,
                     d[self._vars["cruise"]["start"]]),
            "cruise": (d[self._vars["cruise"]["start"]],
                       d[self._vars["fini"]["start"]]),
            "fini": (d[self._vars["fini"]["start"]],
                     self.mutation.interval.stop)
        }

        t = sympy.Symbol("t")
        expression = sympy.Piecewise(*(
            (x_expressions[phase].subs(self._vars[phase]['t'], t),
             t <= phase_boundaries[phase][1]) for phase in phase_order))

        self._func = sympy.lambdify((t, ), expression)

    def x(self, t):
        # `float` to adapt from numpy 1-value array to actual float
        ret = float(self._func(t))
        if np.isnan(ret):
            # ... this happens if we're too near the edge.
            array = np.asarray([self.interval.start, self.interval.stop])
            idx = (np.abs(array - t)).argmin()
            while np.isnan(ret):
                t = np.nextafter(t, array[1 - idx])
                ret = float(self._func(t))
        return ret


class MutationImplFaster:
    def __init__(self, mutation):
        self.mutation = mutation

    def create_equations(self):
        pass

    def supply(self):
        self.solve()

    def solve(self):
        self.x0 = self.mutation.old
        self.x3 = self.mutation.new

        self.t0 = self.mutation.interval.start
        self.t3 = self.mutation.interval.stop

        self.tc = 0.5

        self.t1 = ((self.t3-self.t0)*(1-self.tc))/2 + self.t0
        self.t2 = self.t3 - (self.t1-self.t0)

        self.v = 2*(self.x3-self.x0)/((self.t3-self.t0)*(1+self.tc))

        self.a = self.v/(self.t1-self.t0)

    def x(self, t):
        if t <= self.t0:
            return self.x0
        if t <= self.t1:
            return self.x0 + 0.5 * self.a * (t - self.t0)**2
        if t <= self.t2:
            return self.x(self.t1) + (t - self.t1) * self.v
        if t < self.t3:
            return self.x3 - 0.5 * self.a * (self.t3 - t)**2
        return self.x3


class MutationImplFast:
    def __init__(self, mutation):
        self.mutation = mutation

    def create_equations(self):
        pass

    def supply(self):
        self.solve()

    def solve(self):
        self.x0 = self.mutation.old
        self.x3 = self.mutation.new
        self.x1 = self.x0 + self.mutation.delta / 6
        self.x2 = self.x1 + 2 * self.mutation.delta / 3
        assert np.isclose(self.x3 - self.x2, self.x1 - self.x0)
        assert np.isclose((self.x3 - self.x2)*4, self.x2 - self.x1)

        self.t0 = self.mutation.interval.start
        self.t3 = self.mutation.interval.stop
        self.t1 = self.t0 + self.mutation.interval.duration / 3
        self.t2 = self.t1 + self.mutation.interval.duration / 3

        self.vc = (
            (2 * self.mutation.delta) / (self.mutation.interval.duration))

        assert np.isclose((self.x2 - self.x1), (self.t2 - self.t1) * self.vc)

        self.a = (3 * self.mutation.delta) / (
             (self.mutation.interval.duration**2))

        assert np.isclose(
            self.x(self.t0), self.x(np.nextafter(self.t0, self.t0 + 1)))
        assert np.isclose(
            self.x(self.t1), self.x(np.nextafter(self.t1, self.t1 + 1)))
        assert np.isclose(
            self.x(self.t1), self.x(np.nextafter(self.t1, self.t1 - 1)))
        assert np.isclose(
            self.x(self.t2), self.x(np.nextafter(self.t2, self.t2 + 1)))
        assert np.isclose(
            self.x(self.t2), self.x(np.nextafter(self.t2, self.t2 - 1)))
        assert np.isclose(
            self.x(self.t3), self.x(np.nextafter(self.t3, self.t3 - 1)))

    def x(self, t):
        if t <= self.t0:
            return self.x0
        if t < self.t1:
            return self.x0 + 0.5 * self.a * (t - self.t0)**2
        if t == self.t1:
            return self.x1
        if t < self.t2:
            return self.x1 + (t - self.t1) * self.vc
        if t == self.t2:
            return self.x2
        if t < self.t3:
            return (self.x2 + self.vc * (t - self.t2) +
                    3 * 0.5 * -self.a * (t - self.t2)**2)
        return self.x3


class MutationImplZeroDuration:
    def __init__(self, mutation):
        self.mutation = mutation

    def create_equations(self):
        pass

    def supply(self):
        pass

    def solve(self):
        pass

    def x(self, t):
        if t < self.mutation.interval.start:
            return self.mutation.old
        return self.mutation.new


class Mutation:
    """
    Reifies the operation of changing the value of a certain field of a certai
    actor, including the following data

    - the 'old' value of the field
    - the a 'new' value of the field,
    - begins at absolute time 'start' and  ends at absolute time 'stop'
    - may be an unbound state, at which the 'start' and 'stop' are undefined

    """

    def __init__(self, old, new, mutation_impl=None):
        super().__init__()
        self.old = old
        self.new = new
        self.interval = None
        if mutation_impl is None:
            mutation_impl = MutationImplFaster
        self._impl_obj = mutation_impl(self)

        global MUTATION_COUNTER
        self._id = MUTATION_COUNTER
        MUTATION_COUNTER += 1
        self._impl_obj.create_equations()

    def x(self, t):
        if not self.interval.within(t):
            return None
        if self.is_noop():
            return float(self.new)

        return self._impl_obj.x(t)

    @property
    def delta(self):
        return self.new - self.old

    @property
    def duration(self):
        return self.interval.duration

    def is_noop(self):
        return self.new == self.old

    def unbound(self):
        return self.interval is None

    def bound(self):
        return not self.unbound()

    def supply(self, interval):
        if interval is None:
            raise ValueError("Interval is None?!")
        if self.bound():
            raise ValueError("Mutation is already time bound")
        self.interval = interval

        if np.isclose(self.interval.duration, 0):
            # Special case to prevent division by zero
            self._impl_obj = MutationImplZeroDuration(self)
        self._impl_obj.supply()

    def __repr__(self):
        extra = ""
        if self.bound():
            extra = ", start = {}, stop = {}".format(self.interval.start,
                                                     self.interval.stop)
        return "{}(old = {}, new = {}{})".format(
            type(self).__name__, self.old, self.new, extra)


class RealMutation(Mutation):
    """ A mutation for changing a slot with type real. """
    pass


class IntegerMutation(Mutation):
    """
    A mutation for changing a slot with type integer.
    Only gives integer results.
    """

    def x(self, t):
        if not self.interval.within(t):
            return None
        value = super().x(t)
        return round(value)


class InstantMutation(Mutation):
    """ A mutation that changes from old to new at a certain instant during the
    middle of the period """

    def __init__(self, old, new, instant):
        super().__init__(0., 1.)
        self._raw_old = old
        self._raw_new = new
        self._instant = instant

    def x(self, t):
        value = super().x(t)
        if value is None:
            return None
        return self._raw_new if value >= self._instant else self._raw_old


class StringMutation(InstantMutation):
    """
    A mutation for changing a slot with type string.
    Instantly changes at the middle of the change duration.
    """

    def __init__(self, old, new):
        super().__init__(old, new, 0.5)


class BooleanMutation(InstantMutation):
    """
    A mutation for changing a slot with type boolean.
    Instantly changes at the middle of the change duration.
    """

    def __init__(self, old, new):
        super().__init__(old, new, 0.5)


class ReferenceMutation(InstantMutation):
    """
    A mutation for changing a slot with type reference.
    Instantly changes at the middle of the change duration.
    """

    def __init__(self, old, new):
        super().__init__(old, new, 0.5)
