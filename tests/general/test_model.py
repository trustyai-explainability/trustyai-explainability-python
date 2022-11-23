# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Test model provider interface"""

from common import *
from trustyai.model import Model, feature
from trustyai.utils.data_conversions import numpy_to_prediction_object

import pytest


def test_basic_model():
    """Test basic model"""

    model = Model(lambda x: x, output_names=['a', 'b', 'c', 'd', 'e'])
    features = numpy_to_prediction_object(np.arange(0, 100).reshape(20, 5), feature)
    result = model.predictAsync(features).get()
    assert len(result[0].outputs) == 5


def test_cast_output():
    np2np = Model(lambda x: np.sum(x, 1), output_names=['sum'])
    np2df = Model(lambda x: pd.DataFrame(x))
    df2np = Model(lambda x: x.sum(1).values, dataframe_input=True, output_names=['sum'])
    df2df = Model(lambda x: x, dataframe_input=True)
    pis = numpy_to_prediction_object(np.arange(0., 125.).reshape(25, 5), feature)

    output_val = np2np.predictAsync(pis).get()
    assert len(output_val) == 25

    output_val = np2df.predictAsync(pis).get()
    assert len(output_val) == 25

    output_val = df2np.predictAsync(pis).get()
    assert len(output_val) == 25

    output_val = df2df.predictAsync(pis).get()
    assert len(output_val) == 25


def test_cast_output_arrow():
    np2np = Model(lambda x: np.sum(x, 1), output_names=['sum'], arrow=True)
    np2df = Model(lambda x: pd.DataFrame(x), arrow=True)
    df2np = Model(lambda x: x.sum(1).values, dataframe_input=True, output_names=['sum'], arrow=True)
    df2df = Model(lambda x: x, dataframe_input=True, arrow=True)
    pis = numpy_to_prediction_object(np.arange(0., 125.).reshape(25, 5), feature)

    output_val = np2np.predictAsync(pis).get()
    assert len(output_val) == 25

    output_val = np2df.predictAsync(pis).get()
    assert len(output_val) == 25

    output_val = df2np.predictAsync(pis).get()
    assert len(output_val) == 25

    output_val = df2df.predictAsync(pis).get()
    assert len(output_val) == 25

