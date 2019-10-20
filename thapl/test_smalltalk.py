import unittest

from thapl.context import Compound, Named
from thapl.pattern import Pattern
from thapl.utility import one_or_raise


def create_smalltalk():
    objects = {
        "class": Compound(name=("class", ), actor=True),
        "object": Compound(name=("object", ), actor=True),
        "magnitude": Compound(name=("magnitude", ), actor=True),
        "number": Compound(name=("number", ), actor=True),
        "integer": Compound(name=("integer", ), actor=True),
        "var": Compound(name=("var", ), actor=True)
    }
    # set classes for everything
    objects["class"].append_actor(Named(Pattern(["class"]), objects["class"]))
    objects["object"].append_actor(Named(Pattern(["class"]), objects["class"]))
    objects["magnitude"].append_actor(
        Named(Pattern(["class"]), objects["class"]))
    objects["number"].append_actor(Named(Pattern(["class"]), objects["class"]))
    objects["integer"].append_actor(Named(Pattern(["class"]),
                                          objects["class"]))
    objects["var"].append_actor(Named(Pattern(["class"]), objects["integer"]))

    # set inheritance
    objects["magnitude"].append_actor_inherited(objects["object"])
    objects["number"].append_actor_inherited(objects["magnitude"])
    objects["integer"].append_actor_inherited(objects["number"])

    # set method for example
    objects["object"].append_actor(
        Named(Pattern(["get", "class"]),
              Compound(command=lambda self: self.find_actor(["class"]))))

    return objects


def deref(g):
    return one_or_raise(g, LookupError)[0]


class TestLittleSmalltalk(unittest.TestCase):
    def test_get_class(self):
        smalltalk = create_smalltalk()
        var_class = deref(smalltalk["var"].find_actor(["class"]))
        var_get_class = deref(var_class.find_actor(["get", "class"]))
        self.assertIs(smalltalk["integer"],
                      deref(var_get_class.begin()(smalltalk["var"])))
