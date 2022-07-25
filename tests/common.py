# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys
from typing import List

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import trustyai
import numpy as np
import pandas as pd

from trustyai.model import (
    FeatureFactory,
    PredictionOutput,
    output,
)


def mock_feature(value, name='f-num'):
    """Create a mock numerical feature"""
    return FeatureFactory.newNumericalFeature(name, value)


def sum_skip_model(inputs: np.ndarray) -> np.ndarray:
    """SumSkip test model"""
    return np.sum(inputs[:,[i for i in range(inputs.shape[1]) if i!=5]], 1)
