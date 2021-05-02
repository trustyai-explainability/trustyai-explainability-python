# pylint: disable = import-error, invalid-name
"""General model classes"""
from org.kie.kogito.explainability import (
    TestUtils as _TestUtils,
    Config as _Config,
)
from java.util import ArrayList, List

TestUtils = _TestUtils
Config = _Config


def toJList(pyList):
    """Convert a Python list to a Java ArrayList"""
    result = ArrayList()
    for item in pyList:
        result.add(item)
    return result
