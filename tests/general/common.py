# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys
from typing import Optional, List

import numpy as np
import pandas as pd  # pylint: disable=unused-import

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../src")

from trustyai.model import (
    FeatureFactory,
)


def mock_feature(value, name='f-num'):
    """Create a mock numerical feature"""
    return FeatureFactory.newNumericalFeature(name, value)


def sum_skip_model(inputs: np.ndarray) -> np.ndarray:
    """SumSkip test model"""
    return np.sum(inputs[:, [i for i in range(inputs.shape[1]) if i != 5]], 1)


def create_random_dataframe(weights: Optional[List[float]] = None):
    """Create a simple random Pandas dataframe"""
    from sklearn.datasets import make_classification
    if not weights:
        weights = [0.9, 0.1]

    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=2, class_sep=2, flip_y=0, weights=weights,
                               random_state=23)

    return pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'y': y
    })
