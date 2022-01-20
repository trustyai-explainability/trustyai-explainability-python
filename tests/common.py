# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import trustyai

INITIALISED = False

if not INITIALISED:
    trustyai.init(
        path=trustyai.CORE_DEPS + [
            "./dep/org/optaplanner/optaplanner-core/8.12.0.Final/optaplanner-core-8.12.0.Final.jar",
            "./dep/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar",
            "./dep/org/kie/kie-api/7.59.0.Final/kie-api-7.59.0.Final.jar",
            "./dep/io/micrometer/micrometer-core/1.7.4/micrometer-core-1.7.4.jar",
        ]
    )

    INITIALISED = True

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
