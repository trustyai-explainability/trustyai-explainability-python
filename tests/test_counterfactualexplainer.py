# pylint: disable=import-error, wrong-import-position, wrong-import-order, R0801
"""Test suite for counterfactual explanations"""

from common import *

from java.util import Random
from pytest import approx

from trustyai.explainers import CounterfactualExplainer
from trustyai.model import (
    counterfactual_prediction,
    output, Model, feature,
)
from trustyai.utils import TestUtils

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

    model = TestUtils.getSumSkipModel(0)

    prediction = counterfactual_prediction(
        input_features=features,
        outputs=goal,
    )

    counterfactual_result = explainer.explain(prediction, model)
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

    prediction = counterfactual_prediction(
        input_features=features,
        outputs=goal,
    )
    model = TestUtils.getSumThresholdModel(center, epsilon)
    result = explainer.explain(prediction, model)

    total_sum = 0
    for entity in result._result.entities:
        total_sum += entity.as_feature().value.as_number()
        print(entity)

    print("Counterfactual match:")
    print(result._result.output[0].outputs)

    assert total_sum <= center + epsilon
    assert total_sum >= center - epsilon
    assert result._result.isValid()


def test_counterfactual_match_python_model():
    """Test if there's a valid counterfactual with a Python model"""
    GOAL_VALUE = 1000
    goal = [output(name="sum-but-0", dtype="number", value=GOAL_VALUE, score=1.0)]

    n_features = 5

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0)) for i in range(n_features)
    ]

    explainer = CounterfactualExplainer(steps=1000)

    prediction = counterfactual_prediction(
        input_features=features,
        outputs=goal,
    )

    model = Model(sum_skip_model)

    result = explainer.explain(prediction, model)
    assert sum([entity.as_feature().value.as_number() for entity in result._result.entities]) == approx(GOAL_VALUE, rel=3)


def test_counterfactual_plot():
    """Test if there's a valid counterfactual with a Python model"""
    GOAL_VALUE = 1000
    goal = [output(name="sum-but-0", dtype="number", value=GOAL_VALUE, score=1.0)]

    n_features = 5

    features = [
        feature(name=f"f-num{i + 1}", value=10.0, dtype="number", domain=(0.0, 1000.0)) for i in range(n_features)
    ]

    explainer = CounterfactualExplainer(steps=1000)

    prediction = counterfactual_prediction(
        input_features=features,
        outputs=goal,
    )

    model = Model(sum_skip_model)

    result = explainer.explain(prediction, model)
    result.plot()

def test_counterfactual_v2():
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(1, 5))
    features = [feature(str(k), "number", v, domain=(-10., 10.)) for k, v in data.iloc[0].items()]
    model_weights = np.random.rand(5)
    predict_function = lambda x: np.dot(x.values, model_weights)

    model = Model(predict_function, dataframe=True)
    goal = np.array([[0]])
    prediction = counterfactual_prediction(input_features=features, outputs=goal)
    explainer = CounterfactualExplainer(steps=10_000)
    explanation = explainer.explain(prediction, model)
    result_output = model(explanation.get_proposed_features_as_pandas())
    assert result_output<.01
    assert result_output>-.01