#!/usr/bin/env python3

import os.path

from thapl.parser import TopLevelParser
from thapl.runner import Script
from thapl.then_meanwhile_tree import StringSummarizer


def run(text, cwd):
    parser = TopLevelParser(cwd=cwd)
    obj = parser.parse(text)

    class HookedScript(Script):
        def hook_directive(self, directive):
            print("Directives:")
            print(directive.reduce(StringSummarizer()))
            print()

        def hook_log(self, log):
            print("Prior to springs:")
            print(log.reduce(StringSummarizer()))
            print()

        def hook_compacted(self, log):
            print("Compacted:")
            print(log.reduce(StringSummarizer("spring")))
            print()

        def hook_log_with_solved_springs(self, log):
            print("Spring values:")
            print(log.reduce(StringSummarizer("spring")))
            print()

        def hook_log_with_timings(self, log):
            print("Pretty-printed ThenMeanwhileTree, solved:")
            print(log.reduce(StringSummarizer()))

    script = Script(obj)
    log = script.run()
    return script.render(obj, log)


def main(filename):
    """
    :param filename: Filename to load directive from.

    """
    cwd = os.path.dirname(os.path.realpath(os.path.expanduser(filename)))

    with open(filename, "r") as f:
        text = f.read()

    print(run(text, cwd=cwd))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
