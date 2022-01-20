# pylint: disable=import-error, wrong-import-position, wrong-import-order, R0801
"""Test suite for counterfactual explanations"""

from common import *

import uuid

from trustyai.explainers import (
    CounterfactualExplainer,
    SolverConfigBuilder,
    CounterfactualConfig,
)
from trustyai.utils import TestUtils
from trustyai.model.domain import NumericalFeatureDomain
from trustyai.model import (
    CounterfactualPrediction,
    DataDomain,
    PredictionFeatureDomain,
    PredictionInput,
    FeatureFactory,
    Output,
    PredictionOutput,
    Type,
    Value,
)
from java.util import Random
from java.lang import Long

from org.optaplanner.core.config.solver.termination import TerminationConfig

jrandom = Random()
jrandom.setSeed(0)


def run_counterfactual_search(goal, constraints, data_domain, features, model):
    """Creates a CF explainer and returns a result"""
    termination_config = TerminationConfig().withScoreCalculationCountLimit(
        Long.valueOf(10_000)
    )
    solver_config = SolverConfigBuilder.builder().withTerminationConfig(termination_config).build()
    cf_config = CounterfactualConfig().withSolverConfig(solver_config)
    explainer = CounterfactualExplainer(cf_config)

    input_ = PredictionInput(features)
    output = PredictionOutput(goal)
    domain = PredictionFeatureDomain(data_domain.getFeatureDomains())
    prediction = CounterfactualPrediction(
        input_, output, domain, constraints, None, uuid.uuid4(), Long(60)
    )
    return explainer.explain(prediction, model)


def test_non_empty_input():
    """Checks whether the returned CF entities are not null"""
    termination_config = TerminationConfig().withScoreCalculationCountLimit(
        Long.valueOf(1000)
    )
    solver_config = SolverConfigBuilder.builder().withTerminationConfig(termination_config).build()
    cf_config = CounterfactualConfig().withSolverConfig(solver_config)
    n_features = 10
    explainer = CounterfactualExplainer(cf_config)

    goal = [Output(f"f-num1", Type.NUMBER, Value(10.0), 0.0)]
    features = [
        FeatureFactory.newNumericalFeature(f"f-num{i}", i * 2.0)
        for i in range(n_features)
    ]
    constraints = [False] * n_features
    feature_boundaries = [NumericalFeatureDomain.create(0.0, 1000.0)] * n_features

    model = TestUtils.getSumSkipModel(0)

    prediction = CounterfactualPrediction(
        PredictionInput(features),
        PredictionOutput(goal),
        PredictionFeatureDomain(feature_boundaries),
        constraints,
        None,
        uuid.uuid4(),
        Long(60)
    )

    counterfactual_result = explainer.explain(prediction, model)
    for entity in counterfactual_result.getEntities():
        print(entity)
        assert entity is not None


def test_counterfactual_match():
    """Test if there's a valid counterfactual"""
    goal = [Output("inside", Type.BOOLEAN, Value(True), 0.0)]

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
    for entity in result.getEntities():
        total_sum += entity.asFeature().getValue().asNumber()
        print(entity)

    print(result.getOutput().get(0).getOutputs())

    assert total_sum <= center + epsilon
    assert total_sum >= center - epsilon
    assert result.isValid()
