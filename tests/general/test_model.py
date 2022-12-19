# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Test model provider interface"""
from trustyai.explainers import LimeExplainer

from common import *
from trustyai.model import Model, Dataset, feature

import pytest

from trustyai.utils.data_conversions import numpy_to_prediction_object


def test_basic_model():
    """Test basic model"""

    model = Model(lambda x: x, output_names=['a', 'b', 'c', 'd', 'e'])
    features = numpy_to_prediction_object(np.arange(0, 100).reshape(20, 5), feature)
    result = model.predictAsync(features).get()
    assert len(result[0].outputs) == 5


def test_cast_output():
    np2np = Model(lambda x: np.sum(x, 1), output_names=['sum'], disable_arrow=True)
    np2df = Model(lambda x: pd.DataFrame(x), disable_arrow=True)
    df2np = Model(lambda x: x.sum(1).values,
                  dataframe_input=True,
                  output_names=['sum'],
                  disable_arrow=True)
    df2df = Model(lambda x: x, dataframe_input=True, disable_arrow=True)

    pis = numpy_to_prediction_object(np.arange(0., 125.).reshape(25, 5), feature)

    for m in [np2np, np2df, df2df, df2np]:
        output_val = m.predictAsync(pis).get()
        assert len(output_val) == 25


def test_cast_output_arrow():
    np2np = Model(lambda x: np.sum(x, 1), output_names=['sum'])
    np2df = Model(lambda x: pd.DataFrame(x))
    df2np = Model(lambda x: x.sum(1).values, dataframe_input=True, output_names=['sum'])
    df2df = Model(lambda x: x, dataframe_input=True)
    pis = numpy_to_prediction_object(np.arange(0., 125.).reshape(25, 5), feature)

    for m in [np2np, np2df, df2df, df2np]:
        m._set_arrow(pis[0])
        output_val = m.predictAsync(pis).get()
        assert len(output_val) == 25


def test_error_model(caplog):
    """test that a broken model spits out useful debugging info"""
    m = Model(lambda x: str(x) - str(x))
    try:
        LimeExplainer().explain(0, 0, m)
    except Exception:
        pass

    assert "Fatal runtime error" in caplog.text
    assert "TypeError: unsupported operand type(s) for -: 'str' and 'str'" in caplog.text
