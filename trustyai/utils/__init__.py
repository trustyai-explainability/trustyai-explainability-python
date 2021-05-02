# pylint: disable = import-error
"""General model classes"""
from org.kie.kogito.explainability import (
    TestUtils as _TestUtils,
    Config as _Config,
)
from java.util import ArrayList, List

TestUtils = _TestUtils
Config = _Config


def toJList(pyList):
    result = ArrayList()
    for item in pyList:
        result.add(item)
    return result
