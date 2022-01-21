# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys
from typing import List

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
    PredictionOutput,
    output,
)


def mock_feature(value):
    """Create a mock numerical feature"""
    return FeatureFactory.newNumericalFeature("f-num", value)


from org.kie.kogito.explainability.model import PredictionInput, PredictionOutput


def sum_skip_model(inputs: List[PredictionInput]) -> List[PredictionOutput]:
    """SumSkip test model"""
    features = inputs[0].features
    result = 0.0
    for i in range(len(features)):
        if i != 0:
            result += features[i].value.as_number()
    _output = [output(name="sum-but-0", dtype="number", value=result)]
    return [PredictionOutput(_output)]
