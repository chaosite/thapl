from collections import namedtuple

FindOptions = namedtuple(
    'FindOptions', ['search_environment', 'search_variables', 'visited_envs'],
    defaults=[True, True, frozenset()])
FindMetadata = namedtuple('FindMetadata', 'patterns, receiver, named')
