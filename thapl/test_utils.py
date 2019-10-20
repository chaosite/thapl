""" Some mocks to use for testing """
from thapl import parser

p = None

def tokenize(s):
    global p
    if p is None:
        p = parser.TopLevelParser()
    return list(p.lex(s))
