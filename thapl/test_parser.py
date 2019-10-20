import unittest

from thapl import parser


class TestParser(unittest.TestCase):
    def test_structure(self):
        p = parser.TopLevelParser()
        test = """
play whatever
  act other
    scene i dont care
      action
        do
"""
        p._parser.parse(test)

    def test_decimal_numbers(self):
        p = parser.TopLevelParser()
        test = r"""
play over nine thousand
  action
    change something to four score and seven
"""
        tokens = p._parser.parse(test)
        self.assertEqual(9000, int(tokens.children[0].children[0].children[1]))
        self.assertEqual(
            87,
            int(tokens.children[1].children[0].children[0].children[0].
                children[3].children[0]))

    def test_roman_numerals(self):
        p = parser.TopLevelParser()
        test = r"""
play over MMMMMMMMM
  action
    change something to LXXXVII
"""
        tokens = p._parser.parse(test)
        self.assertEqual(9000, int(tokens.children[0].children[0].children[1]))
        self.assertEqual(
            87,
            int(tokens.children[1].children[0].children[0].children[0].
                children[3].children[0]))

    def test_fractions(self):
        p = parser.TopLevelParser()
        test = r"""
play over two thirds
  action
    change something to five fifths
"""
        tokens = p._parser.parse(test)
        self.assertAlmostEqual(
            2.0 / 3, float(tokens.children[0].children[0].children[1]))
        self.assertAlmostEqual(
            1,
            float(tokens.children[1].children[0].children[0].children[0].
                  children[3].children[0]))

    def test_can_property(self):
        p = parser.TopLevelParser()
        tokens = p._parser.parse(r"""
play can property
  actors
    actor:
      can dance:
        { k = 1, length = 10 }
        [ over the rainbow ]
        relax
""")
        self.assertNotEqual(tokens, None)


class TestTransformer(unittest.TestCase):
    def test_expression(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play good name
  action
    change thing to (( 1.0 + 5.0 ))""")
        self.assertEqual(tree.value.command.children[0]._value[0].value.value,
                         6.0)

    def test_expression_with_reference(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play good name
  actors
    actor:
      has n = 5.5
  action
    change actor\n to (( actor\n + 2.25 ))""")
        self.assertAlmostEqual(
            tree.value.command.children[0]._value[0].value.value, 7.75)

    def test_expression_with_promotion(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play good name
  actors
    actor:
      has n = 5.5
  action
    change actor\n to (( actor\n + 2.25 + 1 ))""")
        self.assertAlmostEqual(
            tree.value.command.children[0]._value[0].value.value, 8.75)

    def test_expression_complex(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play good name
  actors
    actor:
      has n = 5.5
      has m = 3
  action
    change actor\n to (( actor\n + 2.25 * ((actor\m / 2.3) * (((1-2)))) ))""")
        self.assertAlmostEqual(
            tree.value.command.children[0]._value[0].value.value, 2.5652173913)

    def test_expression_lazy(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play good name
  actors
    actor:
      has n = 5.5
      has w = 2
      has m => (( w + 1 ))
  action
    change actor\n to (( actor\n + 2.25 * ((actor\m / 2.3) * (((1-2)))) ))""")
        self.assertAlmostEqual(
            tree.value.command.children[0]._value[0].value.value, 2.5652173913)
        tree.value.actors[0].value.actors[1].value.set_value(4)
        self.assertAlmostEqual(
            tree.value.command.children[0]._value[0].value.value, 0.6086956521)

    def test_expression_lazy_with_cast(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play good name
  actors
    actor:
      has n = 5.5
      has w = 2
      has m (integer) => (( w + 1 ))
  action
    change actor\n to (( actor\n + 2.25 * ((actor\m / 2.3) * (((1-2)))) ))""")
        self.assertAlmostEqual(
            tree.value.command.children[0]._value[0].value.value, 2.5652173913)
        tree.value.actors[0].value.actors[1].value.set_value(4)
        # TODO: This should be -20?
        self.assertAlmostEqual(
            tree.value.command.children[0]._value[0].value.value, 0.6086956521)

    def test_expression_ternary(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play whatever
  actors
    actor:
      has s = (( "foo" ))
      has t = "bar"
      has x => (( true ? s : t ))""")
        self.assertEqual(
            tree.value.actors[0].value.actors[2].value.value.value, "foo")
        tree.value.actors[0].value.actors[0].value._value = "foobar"
        self.assertEqual(
            tree.value.actors[0].value.actors[2].value.value.value, "foobar")

    def test_sub_actors(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play test
  actors
    upper:
      lower:
        has s = "string"
""")
        self.assertEqual(
            tree.value.actors[0].value.actors[0].value.actors[0].value.value,
            "string")

    def test_ctor_no_inheritance(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play test
  actors
    parent <n (integer)> :
      has number => n
    child (parent <4>):
      has other number => (( number ))
""")
        self.assertEqual(
            tree.value.actors[0].value.actors[0].value.value.value, 4)

    def test_ctor_inheritance(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play test
  actors
    grandparent:
      has thing = "foo"
    parent (grandparent) <n (integer)>:
      has number => n
    child (parent <4>):
      has other number => (( number ))
""")
        self.assertEqual(
            tree.value.actors[0].value.actors[0].value.value.value, 4)

    def test_has_meta(self):
        p = parser.TopLevelParser()
        tree = p.parse(r"""
play test
  actors
    thing:
      has meta k = 3
""")
        self.assertTrue(
            tree.value.actors[0].value.actors[0].is_meta)
