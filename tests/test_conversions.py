# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Implicit conversion test suite"""

from common import *

from jpype import _jclass

from trustyai.model.domain import feature_domain


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


def test_numeric_domain_tuple():
    """Test create numeric domain from tuple"""
    domain = (0, 1000)
    jdomain = feature_domain(domain)
    assert jdomain.getLowerBound() == 0
    assert jdomain.getUpperBound() == 1000

    domain = (0.0, 1000.0)
    jdomain = feature_domain(domain)
    assert jdomain.getLowerBound() == 0.0
    assert jdomain.getUpperBound() == 1000.0


def test_empty_domain():
    """Test empty domain"""
    domain = feature_domain(None)
    assert domain.isEmpty() is True


def test_categorical_domain_tuple():
    """Test create categorical domain from tuple and list"""
    domain = ("foo", "bar", "baz")
    jdomain = feature_domain(domain)
    assert jdomain.getCategories().size() == 3
    assert jdomain.getCategories().containsAll(list(domain))

    domain = ["foo", "bar", "baz"]
    jdomain = feature_domain(domain)
    assert jdomain.getCategories().size() == 3
    assert jdomain.getCategories().containsAll(domain)
