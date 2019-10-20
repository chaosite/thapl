#!/usr/bin/env python3


class __Singleton:
    pass


def one_or_raise(iterator, exception):
    singleton = __Singleton()
    ret = next(iterator, singleton)
    if ret is singleton:
        raise exception
    if next(iterator, singleton) is not singleton:
        raise exception
    return ret


def first_or_raise(iterator, exception):
    singleton = __Singleton()
    ret = next(iterator, singleton)
    if ret is singleton:
        raise exception
    return ret
