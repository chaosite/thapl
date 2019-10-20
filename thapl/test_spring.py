import unittest

from thapl import spring


class TestInterval(unittest.TestCase):
    def test_sanity(self):
        start = 0.5
        end = 0.7
        t = spring.Interval(start, end)

        self.assertAlmostEqual(start, t.start)
        self.assertAlmostEqual(end, t.stop)
        self.assertAlmostEqual(end - start, t.duration)

    def test_within(self):
        start = 0.73
        stop = 6.32
        t = spring.Interval(start, stop)

        self.assertTrue(t.within(3.))
        self.assertFalse(t.within(0.1))
        self.assertFalse(t.within(6.34))


class TestSpring(unittest.TestCase):
    def test_sanity(self):
        s = spring.Spring(10, 3, 1)
        s.stretch(spring.Interval(3, 8))
        self.assertAlmostEquals(5, s.interval.duration)

    def test_two_parallel_same_k(self):
        s1 = spring.Spring(5, 1)
        s2 = spring.Spring(10, 1)
        se = spring.ParallelSpring((s1, s2))
        se.solve()

        self.assertAlmostEqual(s1.interval.duration, 7.5)
        self.assertAlmostEqual(s2.interval.duration, 7.5)
        self.assertAlmostEqual(se.interval.duration, 7.5)

    def test_three_parallel_different_k(self):
        s1 = spring.Spring(5, 2)
        s2 = spring.Spring(10, 1)
        s3 = spring.Spring(20, 5)
        se = spring.ParallelSpring((s1, s2, s3))
        se.solve()

        self.assertAlmostEqual(s1.interval.duration, 15)
        self.assertAlmostEqual(s2.interval.duration, 15)
        self.assertAlmostEqual(s3.interval.duration, 15)
        self.assertAlmostEqual(se.interval.duration, 15)

    def test_two_serial_same_k(self):
        s1 = spring.Spring(5, 1)
        s2 = spring.Spring(10, 1)
        se = spring.SerialSpring((s1, s2))
        se.solve()

        self.assertAlmostEqual(s1.interval.start, 0)
        self.assertAlmostEqual(s1.interval.stop, 5)
        self.assertAlmostEqual(s2.interval.start, 5)
        self.assertAlmostEqual(s2.interval.stop, 15)
        self.assertAlmostEqual(s1.interval.duration, 5)
        self.assertAlmostEqual(s2.interval.duration, 10)

    def test_three_serial_different_k(self):
        s1 = spring.Spring(5, 1)
        s2 = spring.Spring(10, 50)
        s3 = spring.Spring(15, 3)
        se = spring.SerialSpring((s1, s2, s3))
        se.solve()

        self.assertAlmostEqual(s1.interval.start, 0)
        self.assertAlmostEqual(s1.interval.stop, 5)
        self.assertAlmostEqual(s2.interval.start, 5)
        self.assertAlmostEqual(s2.interval.stop, 15)
        self.assertAlmostEqual(s3.interval.start, 15)
        self.assertAlmostEqual(s3.interval.stop, 30)
        self.assertAlmostEqual(s1.interval.start, se.interval.start)
        self.assertAlmostEqual(s3.interval.stop, se.interval.stop)

    def test_spring_system(self):
        s_s1 = spring.Spring(10, 1)
        s_s2 = spring.Spring(5, 0.5)
        s_p = spring.Spring(25, 1 / 3)

        s_s = spring.SerialSpring((s_s1, s_s2))
        se = spring.ParallelSpring((s_s, s_p))
        se.solve()

        self.assertAlmostEqual(se.interval.duration, 20)
        self.assertAlmostEqual(s_s1.interval.start, 0)
        self.assertAlmostEqual(s_s1.interval.stop, 10 + 1 + 2/3)
        self.assertAlmostEqual(s_s2.interval.start, 10 + 1 + 2/3)
        self.assertAlmostEqual(s_s2.interval.stop, 20)
