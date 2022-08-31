# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys
from typing import List

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../../")

import numpy as np
import pandas as pd
from trustyai.model import (
    FeatureFactory,
    PredictionOutput,
    output,
)


def mock_feature(value, name="f-num"):
    """Create a mock numerical feature"""
    return FeatureFactory.newNumericalFeature(name, value)


def sum_skip_model(inputs: np.ndarray) -> np.ndarray:
    """SumSkip test model"""
    return np.sum(inputs[:, [i for i in range(inputs.shape[1]) if i != 5]], 1)


import pytest

# pylint: disable=import-error, wrong-import-position, wrong-import-order, R0801
"""Test suite for counterfactual explanations"""

from java.util import Random
from pytest import approx

from trustyai.explainers import CounterfactualExplainer
from trustyai.model import (
    output,
    Model,
    feature,
)
from trustyai.utils import TestUtils

jrandom = Random()
jrandom.setSeed(0)


def non_empty_input():
    """Checks whether the returned CF entities are not null"""
    n_features = 10
    explainer = CounterfactualExplainer(steps=1000)

    goal = [output(name="f-num1", dtype="number", value=10.0, score=0.0)]
    features = [
        feature(name=f"f-num{i}", value=i * 2.0, dtype="number", domain=(0.0, 1000.0))
        for i in range(n_features)
    ]

    model = TestUtils.getSumSkipModel(0)

    counterfactual_result = explainer.explain(inputs=features, goal=goal, model=model)
    for entity in counterfactual_result._result.entities:
        print(entity)
        assert entity is not None


def counterfactual_match():
    """Test if there's a valid counterfactual"""
    goal = [output(name="inside", dtype="bool", value=True, score=0.0)]

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0))
        for i in range(4)
    ]

    center = 500.0
    epsilon = 10.0

    explainer = CounterfactualExplainer(steps=10000)

    model = TestUtils.getSumThresholdModel(center, epsilon)
    result = explainer.explain(inputs=features, goal=goal, model=model)

    total_sum = 0
    for entity in result._result.entities:
        total_sum += entity.as_feature().value.as_number()
        print(entity)

    print("Counterfactual match:")
    print(result._result.output[0].outputs)

    assert total_sum <= center + epsilon
    assert total_sum >= center - epsilon
    assert result._result.isValid()


def counterfactual_match_python_model():
    """Test if there's a valid counterfactual with a Python model"""
    GOAL_VALUE = 1000
    goal = np.array([[GOAL_VALUE]])
    n_features = 5

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0))
        for i in range(n_features)
    ]
    explainer = CounterfactualExplainer(steps=1000)

    model = Model(sum_skip_model, dataframe=False, output_names=["sum-but-5"])

    result = explainer.explain(inputs=features, goal=goal, model=model)

    assert sum(
        [entity.as_feature().value.as_number() for entity in result._result.entities]
    ) == approx(GOAL_VALUE, rel=3)


def counterfactual_plot():
    """Test if there's a valid counterfactual with a Python model"""
    GOAL_VALUE = 1000
    goal = np.array([[GOAL_VALUE]])
    n_features = 5

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0))
        for i in range(n_features)
    ]
    explainer = CounterfactualExplainer(steps=1000)

    model = Model(sum_skip_model, dataframe=False, output_names=["sum-but-5"])

    result = explainer.explain(inputs=features, goal=goal, model=model)

    result.plot()


def counterfactual_v2():
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(1, 5))
    features = [
        feature(str(k), "number", v, domain=(-10.0, 10.0))
        for k, v in data.iloc[0].items()
    ]
    model_weights = np.random.rand(5)
    predict_function = lambda x: np.dot(x.values, model_weights)

    model = Model(predict_function, dataframe=True)
    goal = np.array([[0]])
    explainer = CounterfactualExplainer(steps=10_000)
    explanation = explainer.explain(inputs=features, goal=goal, model=model)
    result_output = model(explanation.proposed_features_dataframe)
    assert result_output < 0.01
    assert result_output > -0.01


def test_fib_10(benchmark):
    benchmark(counterfactual_match)
