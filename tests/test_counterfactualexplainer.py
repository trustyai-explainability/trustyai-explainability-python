# pylint: disable=import-error, wrong-import-position, wrong-import-order, R0801
"""Test suite for counterfactual explanations"""

from java.util import Random

from trustyai.explainers import CounterfactualExplainer
from trustyai.local.counterfactual import counterfactual_prediction
from trustyai.model import (
    FeatureFactory,
    output,
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
        FeatureFactory.newNumericalFeature(f"f-num{i}", i * 2.0)
        for i in range(n_features)
    ]
    constraints = [False] * n_features
    domains = [(0.0, 1000.0)] * n_features

    model = TestUtils.getSumSkipModel(0)

    prediction = counterfactual_prediction(
        input_features=features,
        outputs=goal,
        domains=domains,
        constraints=constraints
    )

    counterfactual_result = explainer.explain(prediction, model)
    for entity in counterfactual_result.entities:
        print(entity)
        assert entity is not None


def test_counterfactual_match():
    """Test if there's a valid counterfactual"""
    goal = [output(name="inside", dtype="bool", value=True, score=0.0)]

    features = [
        FeatureFactory.newNumericalFeature(f"f-num{i + 1}", 10.0) for i in range(4)
    ]
    constraints = [False] * 4
    domains = [(0.0, 1000.0)] * 4

    center = 500.0
    epsilon = 10.0

    explainer = CounterfactualExplainer()

    prediction = counterfactual_prediction(
        input_features=features,
        outputs=goal,
        domains=domains,
        constraints=constraints
    )
    model = TestUtils.getSumThresholdModel(center, epsilon)
    result = explainer.explain(prediction, model)

    total_sum = 0
    for entity in result.entities:
        total_sum += entity.as_feature().value.as_number()
        print(entity)

    print("Counterfactual match:")
    print(result.output[0].outputs)

    assert total_sum <= center + epsilon
    assert total_sum >= center - epsilon
    assert result.isValid()
