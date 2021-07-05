# pylint: disable=R0801
"""Common methods and models for tests"""
from trustyai.model import (
    FeatureFactory,
    Output,
    PredictionOutput,
    Type,
    Value,
)


def mock_feature(value):
    """Create a mock numerical feature"""
    return FeatureFactory.newNumericalFeature("f-num", value)


def sum_skip_model(inputs):
    """SumSkip test model"""
    prediction_outputs = []
    for prediction_input in inputs:
        features = prediction_input.getFeatures()
        result = 0.0
        for i in range(features.size()):
            if i != 0:
                result += features.get(i).getValue().asNumber()
        output = [Output("sum-but0", Type.NUMBER, Value(result), 1.0)]
        prediction_output = PredictionOutput(output)
        prediction_outputs.append(prediction_output)
    return prediction_outputs
