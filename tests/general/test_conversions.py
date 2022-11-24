# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Implicit conversion test suite"""
from common import *

from jpype import _jclass

from trustyai.model import feature
from trustyai.model.domain import feature_domain
from trustyai.utils.data_conversions import (
    one_input_convert,
    one_output_convert,
    many_inputs_convert,
    many_outputs_convert
)
from org.kie.trustyai.explainability.model import Type



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


def test_feature_function():
    """Test helper method to create features"""
    f1 = feature(name="f-1", value=1.0, dtype="number")
    assert f1.name == "f-1"
    assert f1.value.as_number() == 1.0
    assert f1.type == Type.NUMBER

    f2 = feature(name="f-2", value=True, dtype="bool")
    assert f2.name == "f-2"
    assert f2.value.as_obj() == True
    assert f2.type == Type.BOOLEAN

    f3 = feature(name="f-3", value="foo", dtype="categorical")
    assert f3.name == "f-3"
    assert f3.value.as_string() == "foo"
    assert f3.type == Type.CATEGORICAL

    f4 = feature(name="f-4", value=5, dtype="categorical")
    assert f4.name == "f-4"
    assert f4.value.as_number() == 5
    assert f4.type == Type.CATEGORICAL


def test_feature_domains():
    """Test domains"""
    f1 = feature(name="f-1", value=1.0, dtype="number")
    assert f1.name == "f-1"
    assert f1.value.as_number() == 1.0
    assert f1.type == Type.NUMBER
    assert f1.domain is None
    assert f1.is_constrained

    f2 = feature(name="f-2", value=2.0, dtype="number", domain=(0.0, 10.0))
    assert f2.name == "f-2"
    assert f2.value.as_number() == 2.0
    assert f2.type == Type.NUMBER
    assert f2.domain
    print(f2.domain)
    assert not f2.is_constrained


def test_one_input_conversion():
    numpy1 = np.arange(0, 10)
    numpy2 = np.arange(0, 10).reshape(1, 10)
    series = pd.Series(numpy1, index=["input-{}".format(i) for i in range(10)])
    df = pd.DataFrame(numpy2, columns=["input-{}".format(i) for i in range(10)])

    ta_numpy1 = one_input_convert(numpy1)
    ta_numpy2 = one_input_convert(numpy2)
    ta_series = one_input_convert(series)
    ta_df = one_input_convert(df)

    assert ta_numpy1.equals(ta_numpy2)
    assert ta_numpy2.equals(ta_series)
    assert ta_series.equals(ta_df)


def test_one_output_conversion():
    numpy1 = np.arange(0, 10)
    numpy2 = np.arange(0, 10).reshape(1, 10)
    series = pd.Series(numpy1, index=["output-{}".format(i) for i in range(10)])
    df = pd.DataFrame(numpy2, columns=["output-{}".format(i) for i in range(10)])

    ta_numpy1 = one_output_convert(numpy1)
    ta_numpy2 = one_output_convert(numpy2)
    ta_series = one_output_convert(series)
    ta_df = one_output_convert(df)

    assert ta_numpy1.equals(ta_numpy2)
    assert ta_numpy2.equals(ta_series)
    assert ta_series.equals(ta_df)


def test_many_outputs_conversion():
    numpy1 = np.arange(0, 10)
    numpy2 = np.arange(0, 10).reshape(1, 10)
    df = pd.DataFrame(numpy2, columns=["output-{}".format(i) for i in range(10)])

    ta_numpy1 = many_outputs_convert(numpy1)
    ta_numpy2 = many_outputs_convert(numpy2)
    ta_df = many_outputs_convert(df)

    for i in range(1):
        assert ta_numpy1[i].equals(ta_numpy2[i])
        assert ta_numpy2[i].equals(ta_df[i])

def test_many_outputs_conversion2():
    numpy1 = np.arange(0, 100).reshape(10, 10)
    df = pd.DataFrame(numpy1, columns=["output-{}".format(i) for i in range(10)])

    ta_numpy1 = many_outputs_convert(numpy1)
    ta_df = many_outputs_convert(df)

    for i in range(10):
        assert ta_numpy1[i].equals(ta_df[i])

def test_many_inputs_conversion():
    numpy1 = np.arange(0, 10)
    numpy2 = np.arange(0, 10).reshape(1, 10)
    df = pd.DataFrame(numpy2, columns=["input-{}".format(i) for i in range(10)])

    ta_numpy1 = many_inputs_convert(numpy1)
    ta_numpy2 = many_inputs_convert(numpy2)
    ta_df = many_inputs_convert(df)

    for i in range(1):
        assert ta_numpy1[i].equals(ta_numpy2[i])
        assert ta_numpy2[i].equals(ta_df[i])

def test_many_inputs_conversion2():
    numpy1 = np.arange(0, 100).reshape(10, 10)
    df = pd.DataFrame(numpy1, columns=["input-{}".format(i) for i in range(10)])

    ta_numpy1 = many_inputs_convert(numpy1)
    ta_df = many_inputs_convert(df)

    for i in range(10):
        assert ta_numpy1[i].equals(ta_df[i])