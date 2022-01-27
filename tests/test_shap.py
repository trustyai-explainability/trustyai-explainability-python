# pylint: disable=import-error, wrong-import-position, wrong-import-order, duplicate-code
"""SHAP explainer test suite"""

from common import *

import pytest

from trustyai.explainers import SHAPExplainer
from trustyai.model import feature, PredictionInput, simple_prediction
from trustyai.utils import TestUtils


def test_no_variance_one_output():
    """Check if the explanation returned is not null"""
    model = TestUtils.getSumSkipModel(0)

    background = [PredictionInput([feature(name="f", value=value, dtype="number") for value in [1.0, 2.0, 3.0]]) for _
                  in
                  range(2)]

    prediction_outputs = model.predictAsync(background).get()

    predictions = [simple_prediction(input_features=background[i].features, outputs=prediction_outputs[i].outputs) for i
                   in
                   range(2)]

    shap_explainer = SHAPExplainer(background=background)

    explanations = [shap_explainer.explain(prediction, model) for prediction in predictions]

    for explanation in explanations:
        for saliency in explanation.getSaliencies():
            for feature_importance in saliency.getPerFeatureImportance():
                assert feature_importance.getScore() == 0.0
