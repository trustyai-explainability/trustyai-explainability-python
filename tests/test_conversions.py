# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Implicit conversion test suite"""
from common import *

from jpype import _jclass


def test_list_python_to_java():
    """Test Python to Java List conversion"""
    python_list = [2, 4, 3, 5, 1]
    minimum = _jclass.JClass('java.util.Collections').min(python_list)
    assert minimum == 1


def test_list_java_to_python():
    """Test Java to Python List conversion"""
    python_list = [2, 4, 3, 5, 1]
    java_list = _jclass.JClass('java.util.Arrays').asList(python_list)
    assert 15 == sum(java_list)
