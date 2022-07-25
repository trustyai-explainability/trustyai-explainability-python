# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Test model provider interface"""

from common import *

from trustyai.model import Model, Dataset, feature


def foo():
    return "works!"


def test_basic_model():
    """Test basic model"""

    model = Model(lambda x: x, output_names=['a','b','c','d','e'])
    features = Dataset.numpy_to_prediction_object(np.arange(0, 100).reshape(20, 5), feature)
    result = model.predictAsync(features).get()
    assert len(result[0].outputs) == 5
