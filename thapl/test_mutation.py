import unittest

from thapl.mutations import Mutation
from thapl.spring import Interval


class TestMutation(unittest.TestCase):
    def test_mutate(self):
        m = Mutation(0., 10.)
        interval = Interval(0, 10)
        m.supply(interval)
        print()
        t1 = m._impl_obj.t1
        t2 = m._impl_obj.t2
        v = m._impl_obj.v
        print(m._impl_obj.t1)
        print(m._impl_obj.t2)
        print(v)
        print(m.x(0))
        print(m.x(t1))
        print(m.x(t2))
        print(m.x(10))
        

        self.assertAlmostEqual(m.x(0), 0)
        self.assertAlmostEqual((m.x(t2) - m.x(t1))/(t2 - t1), v)
        self.assertAlmostEqual(m.x(t1), t1 * v * 0.5)
        self.assertAlmostEqual((m.x(10) - m.x(t2))/(10-t2), v * 0.5)
