# pylint: disable=import-error, wrong-import-position, wrong-import-order, duplicate-code
"""LIME explainer test suite"""
import sys
import os
import pytest
from common import mock_feature

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import trustyai

trustyai.init(
    path=[
        "./dep/org/kie/kogito/explainability-core/1.5.0.Final/*",
        "./dep/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar",
        "./dep/org/apache/commons/commons-lang3/3.8.1/commons-lang3-3.8.1.jar",
    ]
)

DEFAULT_NO_OF_PERTURBATIONS = 1

from trustyai.local.lime import LimeConfig, LimeExplainer
from trustyai.utils import TestUtils, Config
from trustyai.model import (
    PerturbationContext,
    PredictionInput,
    FeatureFactory,
    SimplePrediction,
)
from java.util import Random
from org.kie.kogito.explainability.local import (
    LocalExplanationException,
)

jrandom = Random()
jrandom.setSeed(0)


def test_empty_prediction():
    """Check if the explanation returned is not null"""
    lime_config = (
        LimeConfig()
        .withPerturbationContext(
            PerturbationContext(jrandom, DEFAULT_NO_OF_PERTURBATIONS)
        )
        .withSamples(10)
    )
    lime_explainer = LimeExplainer(lime_config)
    input_ = PredictionInput([])
    model = TestUtils.getSumSkipModel(0)
    output = (
        model.predictAsync([input_])
        .get(Config.INSTANCE.getAsyncTimeout(), Config.INSTANCE.getAsyncTimeUnit())
        .get(0)
    )
    prediction = SimplePrediction(input_, output)
    with pytest.raises(LocalExplanationException):
        lime_explainer.explainAsync(prediction, model)


def test_non_empty_input():
    """Test for non-empty input"""
    lime_config = (
        LimeConfig()
        .withPerturbationContext(
            PerturbationContext(jrandom, DEFAULT_NO_OF_PERTURBATIONS)
        )
        .withSamples(10)
    )
    lime_explainer = LimeExplainer(lime_config)
    features = [FeatureFactory.newNumericalFeature(f"f-num{i}", i) for i in range(4)]

    _input = PredictionInput(features)

    model = TestUtils.getSumSkipModel(0)
    output = model.predictAsync([_input]).get().get(0)
    prediction = SimplePrediction(_input, output)
    saliency_map = lime_explainer.explainAsync(prediction, model).get()
    assert saliency_map is not None


def test_sparse_balance():  # pylint: disable=too-many-locals
    """Test sparse balance"""
    for n_features in range(1, 4):
        no_of_samples = 100
        lime_config_no_penalty = (
            LimeConfig()
            .withPerturbationContext(
                PerturbationContext(jrandom, DEFAULT_NO_OF_PERTURBATIONS)
            )
            .withSamples(no_of_samples)
            .withPenalizeBalanceSparse(False)
        )
        lime_explainer_no_penalty = LimeExplainer(lime_config_no_penalty)

        features = [mock_feature(i) for i in range(n_features)]

        input_ = PredictionInput(features)
        model = TestUtils.getSumSkipModel(0)
        output = model.predictAsync([input_]).get().get(0)
        prediction = SimplePrediction(input_, output)

        saliency_map_no_penalty = lime_explainer_no_penalty.explainAsync(
            prediction, model
        ).get()

        assert saliency_map_no_penalty is not None

        decision_name = "sum-but0"
        saliency_no_penalty = saliency_map_no_penalty.get(decision_name)

        lime_config = (
            LimeConfig().withSamples(no_of_samples).withPenalizeBalanceSparse(True)
        )
        lime_explainer = LimeExplainer(lime_config)

        saliency_map = lime_explainer.explainAsync(prediction, model).get()
        assert saliency_map is not None

        saliency = saliency_map.get(decision_name)

        for i in range(len(features)):
            score = saliency.getPerFeatureImportance().get(i).getScore()
            score_no_penalty = (
                saliency_no_penalty.getPerFeatureImportance().get(i).getScore()
            )
            assert abs(score) <= abs(score_no_penalty)


def test_normalized_weights():
    """Test normalized weights"""
    lime_config = (
        LimeConfig()
        .withNormalizeWeights(True)
        .withPerturbationContext(PerturbationContext(jrandom, 2))
        .withSamples(10)
    )
    lime_explainer = LimeExplainer(lime_config)
    n_features = 4
    features = [mock_feature(i) for i in range(n_features)]
    input_ = PredictionInput(features)
    model = TestUtils.getSumSkipModel(0)
    output = model.predictAsync([input_]).get().get(0)
    prediction = SimplePrediction(input_, output)

    saliency_map = lime_explainer.explainAsync(prediction, model).get()
    assert saliency_map is not None

    decision_name = "sum-but0"
    saliency = saliency_map.get(decision_name)
    per_feature_importance = saliency.getPerFeatureImportance()
    for feature_importance in per_feature_importance:
        assert -1.0 < feature_importance.getScore() < 1.0
