#!/usr/bin/env python3
""" Top-level Thapl constructs """
from thapl.render import Renderer
from thapl.then_meanwhile_tree import (InterpretTransformer,
                                       ExecuteTransformer, SpringAttacher,
                                       TimeSupplier, Compacter)


class Script:
    """
    A Thapl script, representing the action part of a Thapl program (found in a
    scene, act, play, or top-level).  Expects a directive to execute and a
    top-level context for this part of the program.
    """
    def __init__(self, obj):
        self._top_level = obj

    def run(self):
        """
        Perform a dry-run of this script, collecting actions to render later.
        """
        scratch_context = self._top_level  #.clone()
        return self._run(scratch_context)

    def _run(self, obj):
        directive = self._top_level.begin()
        self.hook_directive(directive)
        instruction = directive.reduce(InterpretTransformer(obj.value))
        self.hook_instruction(instruction)
        log = instruction.reduce(ExecuteTransformer(obj.value))
        self.hook_log(log)
        log = log.reduce(Compacter())
        self.hook_compacted(log)
        SpringAttacher().visit(log)
        self.hook_log_with_springs(log)
        log.spring.solve()
        self.hook_log_with_solved_springs(log)
        TimeSupplier().visit(log)
        self.hook_log_with_timings(log)
        return log

    def render(self, context, log):
        """
        Render the action into a list of contexts, discretizing the program.

        :param context: The context describing the environment of the
          instructions.
        :param log: A log of instructions to render.

        """

        return Renderer(log, context).render()

    # Hooks for inheriting classes.

    def hook_directive(self, directive):
        pass

    def hook_instruction(self, instruction):
        pass

    def hook_log(self, log):
        pass

    def hook_compacted(self, log):
        pass

    def hook_log_with_springs(self, log):
        pass

    def hook_log_with_solved_springs(self, log):
        pass

    def hook_log_with_timings(self, log):
        pass
