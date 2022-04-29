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
    trustyai.init()
    INITIALISED = True

from trustyai.model import (
    FeatureFactory,
    PredictionOutput,
    output,
)


def mock_feature(value, name='f-num'):
    """Create a mock numerical feature"""
    return FeatureFactory.newNumericalFeature(name, value)


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
