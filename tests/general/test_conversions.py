# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Implicit conversion test suite"""
from typing import List

from common import *

from jpype import _jclass

from trustyai.model import feature, full_text_feature
from trustyai.model.domain import feature_domain
from trustyai.utils.data_conversions import (
    one_input_convert,
    one_output_convert,
    many_inputs_convert,
    many_outputs_convert, to_trusty_dataframe
)
from org.kie.trustyai.explainability.model import Type

from trustyai.utils import text


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


def test_categorical_numeric_domain_list():
    """Test create numeric domain from list"""
    domain = [0, 1000]
    jdomain = feature_domain(domain)
    assert jdomain.getCategories().size() == 2
    assert jdomain.getCategories().containsAll(domain)

    domain = [0.0, 1000.0]
    jdomain = feature_domain(domain)
    assert jdomain.getCategories().size() == 2
    assert jdomain.getCategories().containsAll(domain)


def test_categorical_object_domain_list():
    """Test create object domain from list"""
    domain = [True, False]
    jdomain = feature_domain(domain)
    assert str(jdomain.getClass().getSimpleName()) == "ObjectFeatureDomain"
    assert jdomain.getCategories().size() == 2
    assert jdomain.getCategories().containsAll(domain)


def test_categorical_object_domain_list_2():
    """Test create object domain from list"""
    domain = [b"test", b"test2"]
    jdomain = feature_domain(domain)
    assert str(jdomain.getClass().getSimpleName()) == "ObjectFeatureDomain"
    assert jdomain.getCategories().size() == 2
    assert jdomain.getCategories().containsAll(domain)


def test_empty_domain():
    """Test empty domain"""
    domain = feature_domain(None)
    assert domain.isEmpty() is True


def test_categorical_domain_tuple():
    """Test create categorical domain from tuple and list"""
    domain = ["foo", "bar", "baz"]
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

    @text.tokenizer
    def tokenizer(x: str) -> List[str]:
        return x.split(" ")

    values = "you just requested to change your password"
    f5 = full_text_feature(name="f-5", value=values, tokenizer=tokenizer)
    assert f5.name == "f-5"
    assert len(f5.value.as_obj()) == 7
    sub_features = f5.value.as_obj()
    tokens = values.split(" ")
    for i in range(7):
        assert sub_features[i].name == "f-5_" + str(i + 1)
        assert sub_features[i].value.as_string() == tokens[i]
    assert f5.type == Type.COMPOSITE


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
    """Test one input conversions to one PredInput"""
    numpy1 = np.arange(0, 10)
    numpy2 = numpy1.reshape(1, 10)

    to_convert = [
        numpy1,
        numpy2,
        pd.Series(numpy1, index=["input-{}".format(i) for i in range(10)]),
        pd.DataFrame(numpy2, columns=["input-{}".format(i) for i in range(10)]),
        numpy1.tolist()
    ]

    converted = [one_input_convert(x) for x in to_convert]

    for i in range(len(converted) - 1):
        assert converted[i].equals(converted[i + 1])


def test_one_input_conversion_domained():
    """Test one input conversions with domains to one PredInput"""
    n_feats = 5
    np.random.seed(0)
    numpy1 = np.arange(0, n_feats)
    numpy2 = numpy1.reshape(1, n_feats)
    domain_bounds = [[np.random.rand(), np.random.rand()] for _ in range(n_feats)]
    domains = [feature_domain((lb, ub)) for lb, ub in domain_bounds]

    to_convert = [
        numpy1,
        numpy2,
        pd.Series(numpy1, index=["input-{}".format(i) for i in range(n_feats)]),
        pd.DataFrame(numpy2, columns=["input-{}".format(i) for i in range(n_feats)]),
        numpy1.tolist()
    ]
    converted = [one_input_convert(x, feature_domains=domains) for x in to_convert]

    for i in range(len(converted) - 1):
        for j in range(n_feats):
            assert converted[i].getFeatures().get(j).getDomain().getLowerBound() \
                   == domain_bounds[j][0]
            assert converted[i].getFeatures().get(j).getDomain().getUpperBound() \
                   == domain_bounds[j][1]

        assert converted[i].equals(converted[i + 1])


def test_one_input_one_feature_conversion():
    """Test one input, one feature conversions to one PredInput"""
    numpy1 = np.arange(0, 1)
    numpy2 = numpy1.reshape(1, 1)

    to_convert = [
        numpy1,
        numpy2,
        pd.Series(numpy1, index=["input-{}".format(i) for i in range(1)]),
        pd.DataFrame(numpy2, columns=["input-{}".format(i) for i in range(1)]),
        numpy1.tolist(),
        numpy1.tolist()[0]
    ]

    converted = [one_input_convert(x) for x in to_convert]

    for i in range(len(converted) - 1):
        assert converted[i].equals(converted[i + 1])


def test_one_output_conversion():
    """Test one output conversions to one PredOutput"""
    numpy1 = np.arange(0, 10)
    numpy2 = numpy1.reshape(1, 10)

    to_convert = [
        numpy1,
        numpy2,
        pd.Series(numpy1, index=["output-{}".format(i) for i in range(10)]),
        pd.DataFrame(numpy2, columns=["output-{}".format(i) for i in range(10)]),
        numpy1.tolist()
    ]

    converted = [one_output_convert(x) for x in to_convert]

    for i in range(len(converted) - 1):
        assert converted[i].equals(converted[i + 1])


def test_one_output_one_value_conversion():
    """Test one output, one value conversions to one PredOutput"""
    numpy1 = np.arange(0, 1)
    numpy2 = numpy1.reshape(1, 1)

    to_convert = [
        numpy1,
        numpy2,
        pd.Series(numpy1, index=["output-{}".format(i) for i in range(1)]),
        pd.DataFrame(numpy2, columns=["output-{}".format(i) for i in range(1)]),
        numpy1.tolist(),
        numpy1.tolist()[0]
    ]

    converted = [one_output_convert(x) for x in to_convert]

    for i in range(len(converted) - 1):
        assert converted[i].equals(converted[i + 1])


def test_many_outputs_conversion():
    """Test many output conversions to PredOutputs, using one row to produce
    List[PredOutputs] with one item"""
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
    """Test many output conversions to many PredOutputs"""
    numpy1 = np.arange(0, 100).reshape(10, 10)
    df = pd.DataFrame(numpy1, columns=["output-{}".format(i) for i in range(10)])

    ta_numpy1 = many_outputs_convert(numpy1)
    ta_df = many_outputs_convert(df)

    for i in range(10):
        assert ta_numpy1[i].equals(ta_df[i])


def test_many_inputs_conversion():
    """Test many input conversions to PredOutputs, using one row to produce
    List[PredInputs] with one item"""
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
    """Test many input conversions to many PredInputs"""
    numpy1 = np.arange(0, 100).reshape(10, 10)
    df = pd.DataFrame(numpy1, columns=["input-{}".format(i) for i in range(10)])

    ta_numpy1 = many_inputs_convert(numpy1)
    ta_df = many_inputs_convert(df)

    for i in range(10):
        assert ta_numpy1[i].equals(ta_df[i])


def test_many_inputs_conversion_domained():
    """Test many input conversions to many PredInputs with domains"""
    n_feats = 5
    n_datapoints = 100
    np.random.seed(0)

    domain_bounds = [[np.random.rand(), np.random.rand()] for _ in range(n_feats)]
    domains = [feature_domain((lb, ub)) for lb, ub in domain_bounds]
    numpy1 = np.arange(0, n_feats * n_datapoints).reshape(-1, n_feats)
    df = pd.DataFrame(numpy1, columns=["input-{}".format(i) for i in range(n_feats)])

    ta_numpy1 = many_inputs_convert(numpy1, feature_domains=domains)
    ta_df = many_inputs_convert(df, feature_domains=domains)

    assert len(ta_numpy1) == n_datapoints
    assert len(ta_df) == n_datapoints

    for converted in [ta_numpy1, ta_df]:
        for i in range(n_datapoints):
            for j in range(n_feats):
                assert converted[i].getFeatures().get(j).getDomain().getLowerBound() \
                       == domain_bounds[j][0]
                assert converted[j].getFeatures().get(j).getDomain().getUpperBound() \
                       == domain_bounds[j][1]

    for i in range(n_datapoints):
        assert ta_numpy1[i].equals(ta_df[i])


def test_numpy_to_trusty_dataframe():
    """Test converting a NumPy array to a TrustyAI dataframe"""
    arr = create_random_dataframe().to_numpy()

    tdf = to_trusty_dataframe(data=arr, feature_names=["x1", "x2", "y"])

    assert tdf.getColumnDimension() == 3
    assert tdf.getRowDimension() == 5000
    assert list(tdf.getColumnNames()) == ["x1", "x2", "y"]
    assert tdf.getInputsCount() == 2
    assert tdf.getOutputsCount() == 1
