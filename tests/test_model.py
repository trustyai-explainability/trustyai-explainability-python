# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Test model provider interface"""

from common import *

from trustyai.model import Model, feature


def foo():
    return "works!"


def test_basic_model():
    """Test basic model"""

    def test_model(inputs):
        outputs = [output(name=feature.name, dtype="number", value=feature.value.as_number()) for feature in
                   inputs]
        return [PredictionOutput(outputs)]

    model = Model(test_model)

    features = [
        feature(name=f"f-num{i}", value=i * 2.0, dtype="number")
        for i in range(5)
    ]

    result = model.predictAsync(features).get()
    assert len(result[0].outputs) == 5
