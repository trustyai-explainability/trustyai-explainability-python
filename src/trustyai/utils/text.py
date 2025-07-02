"""Utility methods for text data handling"""

from typing import List, Callable

from jpype import _jclass


def tokenizer(function: Callable[[str], List[str]]):
    """Post-process outputs of a Python tokenizer function"""

    def wrapper(_input: str):
        return _jclass.JClass("java.util.Arrays").asList(function(_input))

    return wrapper
