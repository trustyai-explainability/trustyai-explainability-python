# pylint: disable=import-error, wrong-import-position, wrong-import-order, R0801
"""Test suite for counterfactual explanations"""

from common import *

import uuid

from java.lang import Long
from java.util import Random

from trustyai.explainers import CounterfactualExplainer
from trustyai.local.counterfactual import counterfactual_prediction
from trustyai.model import (
    CounterfactualPrediction,
    DataDomain,
    PredictionFeatureDomain,
    PredictionInput,
    FeatureFactory,
    PredictionOutput,
    output,
)
from trustyai.model.domain import NumericalFeatureDomain
from trustyai.utils import TestUtils
from org.kie.kogito.explainability.local.counterfactual import CounterfactualResult

jrandom = Random()
jrandom.setSeed(0)


def run_counterfactual_search(goal,
                              constraints,
                              data_domain,
                              features,
                              model,
                              steps=10_000) -> CounterfactualResult:
    """Creates a CF explainer and returns a result"""
    explainer = CounterfactualExplainer(steps=steps)

    input_ = PredictionInput(features)
    output_ = PredictionOutput(goal)
    domain = PredictionFeatureDomain(data_domain.getFeatureDomains())
    prediction = counterfactual_prediction(
        input_=input_,
        output=output_,
        domain=domain,
        constraints=constraints
    )
    return explainer.explain(prediction, model)


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
    feature_boundaries = [NumericalFeatureDomain.create(0.0, 1000.0)] * n_features

    model = TestUtils.getSumSkipModel(0)

    prediction = counterfactual_prediction(
        input_=PredictionInput(features),
        output=PredictionOutput(goal),
        domain=PredictionFeatureDomain(feature_boundaries),
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
    feature_boundaries = [NumericalFeatureDomain.create(0.0, 1000.0)] * 4

    data_domain = DataDomain(feature_boundaries)

    center = 500.0
    epsilon = 10.0

    result = run_counterfactual_search(
        goal,
        constraints,
        data_domain,
        features,
        TestUtils.getSumThresholdModel(center, epsilon),
    )

    total_sum = 0
    for entity in result.entities:
        total_sum += entity.as_feature().value.as_number()
        print(entity)

    print("Counterfactual match:")
    print(result.output[0].outputs)

    assert total_sum <= center + epsilon
    assert total_sum >= center - epsilon
    assert result.isValid()
