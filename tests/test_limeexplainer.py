# pylint: disable=import-error, wrong-import-position, wrong-import-order, duplicate-code
"""LIME explainer test suite"""

from common import *

import pytest

from trustyai.explainers import LimeExplainer
from trustyai.utils import TestUtils
from trustyai.model import feature, simple_prediction

from org.kie.trustyai.explainability.local import (
    LocalExplanationException,
)


def mock_features(n_features: int):
    return [mock_feature(i, f"f-num{i}") for i in range(n_features)]


def test_empty_prediction():
    """Check if the explanation returned is not null"""
    lime_explainer = LimeExplainer(seed=0, samples=10, perturbations=1)
    inputs = []
    model = TestUtils.getSumSkipModel(0)
    outputs = model.predict([inputs])[0].outputs
    prediction = simple_prediction(input_features=inputs, outputs=outputs)
    with pytest.raises(LocalExplanationException):
        lime_explainer.explain(prediction, model)


def test_non_empty_input():
    """Test for non-empty input"""
    lime_explainer = LimeExplainer(seed=0, samples=10, perturbations=1)
    features = [feature(name=f"f-num{i}", value=i, dtype="number") for i in range(4)]

    model = TestUtils.getSumSkipModel(0)
    outputs = model.predict([features])[0].outputs
    prediction = simple_prediction(input_features=features, outputs=outputs)
    saliency_map = lime_explainer.explain(prediction, model)
    assert saliency_map is not None


def test_sparse_balance():  # pylint: disable=too-many-locals
    """Test sparse balance"""
    for n_features in range(1, 4):
        lime_explainer_no_penalty = LimeExplainer(samples=100, penalise_sparse_balance=False)

        features = mock_features(n_features)

        model = TestUtils.getSumSkipModel(0)
        outputs = model.predict([features])[0].outputs
        prediction = simple_prediction(input_features=features, outputs=outputs)

        saliency_map_no_penalty = lime_explainer_no_penalty.explain(
            prediction, model
        ).map()

        assert saliency_map_no_penalty is not None

        decision_name = "sum-but0"
        saliency_no_penalty = saliency_map_no_penalty.get(decision_name)

        lime_explainer = LimeExplainer(samples=100, penalise_sparse_balance=True)

        saliency_map = lime_explainer.explain(prediction, model).map()
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
    lime_explainer = LimeExplainer(normalise_weights=True, perturbations=2, samples=10)
    n_features = 4
    features = mock_features(n_features)
    model = TestUtils.getSumSkipModel(0)
    outputs = model.predict([features])[0].outputs
    prediction = simple_prediction(input_features=features, outputs=outputs)

    saliency_map = lime_explainer.explain(prediction, model).map()
    assert saliency_map is not None

    decision_name = "sum-but0"
    saliency = saliency_map.get(decision_name)
    per_feature_importance = saliency.getPerFeatureImportance()
    for feature_importance in per_feature_importance:
        assert -3.0 < feature_importance.getScore() < 3.0


def test_lime_plots():
    """Test normalized weights"""
    lime_explainer = LimeExplainer(normalise_weights=False, perturbations=2, samples=10)
    n_features = 15
    features = mock_features(n_features)
    model = TestUtils.getSumSkipModel(0)
    outputs = model.predict([features])[0].outputs
    prediction = simple_prediction(input_features=features, outputs=outputs)

    lime_results = lime_explainer.explain(prediction, model)
    lime_results.plot("sum-but0")
