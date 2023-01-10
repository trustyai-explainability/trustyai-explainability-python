# pylint: disable=import-error, wrong-import-position, wrong-import-order, R0801
"""Test suite for counterfactual explanations"""
import math
import random

import pandas as pd
import pytest

from common import *

from java.util import Random
from pytest import approx

from trustyai.explainers import CounterfactualExplainer
from trustyai.explainers.counterfactuals import GoalCriteria
from org.kie.trustyai.explainability.local.counterfactual.goal import GoalScore
from trustyai.model import (
    output, Model, feature,
)
from trustyai.utils import TestModels
from trustyai.model.domain import feature_domain
from trustyai.utils.data_conversions import one_input_convert

jrandom = Random()
jrandom.setSeed(0)


def test_non_empty_input():
    """Checks whether the returned CF entities are not null"""
    n_features = 10
    explainer = CounterfactualExplainer(steps=1000)

    goal = [output(name="f-num1", dtype="number", value=10.0, score=0.0)]
    features = [
        feature(name=f"f-num{i}", value=i * 2.0, dtype="number", domain=(0.0, 1000.0))
        for i in range(n_features)
    ]

    model = TestModels.getSumSkipModel(0)

    counterfactual_result = explainer.explain(
        inputs=features,
        goal=goal,
        model=model)
    for entity in counterfactual_result._result.entities:
        print(entity)
        assert entity is not None


def test_counterfactual_match():
    """Test if there's a valid counterfactual"""
    goal = [output(name="inside", dtype="bool", value=True, score=0.0)]

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0)) for i in range(4)
    ]

    center = 500.0
    epsilon = 10.0

    explainer = CounterfactualExplainer(steps=10000)

    model = TestModels.getSumThresholdModel(center, epsilon)
    result = explainer.explain(
        inputs=features,
        goal=goal,
        model=model)

    total_sum = 0
    for entity in result._result.entities:
        total_sum += entity.as_feature().value.as_number()
        print(entity)

    print("Counterfactual match:")
    print(result._result.output[0].outputs)

    assert total_sum <= center + epsilon
    assert total_sum >= center - epsilon
    assert result._result.isValid()


def test_counterfactual_match_goal_criteria():
    """Test if there's a valid counterfactual using a custom criteria"""

    goal = [output(name="sum-but3", dtype="number", value=0, score=0.0), output(name="sum-but3*2", dtype="number", value=0, score=0.0)]

    def custom_goal(_df):
        f1 = _df['sum-but3'].iloc[0]
        f2 = _df['sum-but3*2'].iloc[0]
        if math.sqrt(f1) == f2:
            return 0, 0
        else:
            return f1-math.sqrt(f2), 1.0

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0)) for i in range(3)
    ]

    explainer = CounterfactualExplainer(steps=20000)
    criteria = GoalCriteria(custom_goal)

    model = TestModels.getSumSkipTwoOutputModel(3)
    result = explainer.explain(
        inputs=features,
        goal=goal,
        model=model,
        criteria=criteria
    )

    total_sum = 0
    for entity in result._result.entities:
        total_sum += entity.as_feature().value.as_number()
        print(entity)

    print("Counterfactual match:")
    print(result._result.output[0].outputs)

    # assert total_sum <= center + epsilon
    # assert total_sum >= center - epsilon
    # assert result._result.isValid()


def test_counterfactual_match_python_model():
    """Test if there's a valid counterfactual with a Python model"""
    GOAL_VALUE = 1000
    goal = np.array([[GOAL_VALUE]])
    n_features = 5

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0)) for i in range(n_features)
    ]
    explainer = CounterfactualExplainer(steps=1000)

    model = Model(sum_skip_model, dataframe_input=False, output_names=['sum-but-5'])

    result = explainer.explain(
        inputs=features,
        goal=goal,
        model=model)

    assert sum([entity.as_feature().value.as_number() for entity in result._result.entities]) == approx(GOAL_VALUE,
                                                                                                        rel=3)


def counterfactual_plot(block):
    """Test if there's a valid counterfactual with a Python model"""
    GOAL_VALUE = 1000
    goal = np.array([[GOAL_VALUE]])
    n_features = 5

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0)) for i in range(n_features)
    ]
    explainer = CounterfactualExplainer(steps=1000)

    model = Model(sum_skip_model, dataframe_input=False, output_names=['sum-but-5'])

    result = explainer.explain(
        inputs=features,
        goal=goal,
        model=model)

    result.plot(block=block)


@pytest.mark.block_plots
def test_counterfactual_plot_blocking():
    counterfactual_plot(True)


def test_counterfactual_plot():
    counterfactual_plot(False)


def test_counterfactual_v2():
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(1, 5))
    features = [feature(str(k), "number", v, domain=(-10., 10.)) for k, v in data.iloc[0].items()]
    model_weights = np.random.rand(5)
    predict_function = lambda x: np.dot(x.values, model_weights)

    model = Model(predict_function, dataframe_input=True)
    goal = np.array([[0]])
    explainer = CounterfactualExplainer(steps=10_000)
    explanation = explainer.explain(
        inputs=features,
        goal=goal,
        model=model)
    result_output = model(explanation.proposed_features_dataframe)
    assert result_output < .01
    assert result_output > -.01


def test_counterfactual_with_domain_argument():
    """Test passing domains to counterfactuals"""
    np.random.seed(0)
    data = np.random.rand(1, 5)
    model_weights = np.random.rand(5)
    model = Model(lambda x: np.dot(x, model_weights))
    explainer = CounterfactualExplainer(steps=10_000)
    explanation = explainer.explain(
        inputs=data,
        goal=np.array([0]),
        feature_domains=[feature_domain((-10, 10)) for _ in range(5)],
        model=model)
    result_output = model(explanation.proposed_features_dataframe)
    assert result_output < .01
    assert result_output > -.01


def test_counterfactual_with_domain_argument_overwrite():
    """Test that passing domains to counterfactuals with already-domained features throws
     a warning"""
    np.random.seed(0)
    data = np.random.rand(1, 5)
    domained_inputs = one_input_convert(data, feature_domains=[feature_domain((-10, 10)) for _ in range(5)])
    model_weights = np.random.rand(5)
    model = Model(lambda x: np.dot(x, model_weights))
    explainer = CounterfactualExplainer(steps=10_000)

    with pytest.warns(UserWarning):
        explainer.explain(
            inputs=domained_inputs,
            goal=np.array([0]),
            feature_domains=[feature_domain((-10, 10)) for _ in range(5)],
            model=model
        )
