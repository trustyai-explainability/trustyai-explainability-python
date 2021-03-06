# pylint: disable=import-error, wrong-import-position, wrong-import-order, duplicate-code
"""SHAP explainer test suite"""

from common import *

import pandas as pd
import numpy as np

np.random.seed(0)

import pytest

from trustyai.explainers import SHAPExplainer
from trustyai.model import feature, PredictionInput, simple_prediction, ArrowModel
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

@pytest.mark.skip(reason="https://issues.redhat.com/browse/FAI-785")
def test_shap_arrow():
    data = pd.DataFrame(np.random.rand(101, 5))
    background = [PredictionInput([feature(name=str(k), value=value, dtype="number") for k, value in row.items()]) for idx, row in data.iloc[:100].iterrows()]
    toExplain  = [PredictionInput([feature(name=str(k), value=value, dtype="number") for k, value in row.items()]) for idx, row in data.iloc[100:].iterrows()]

    model_weights = np.random.rand(5)
    pandas_predict_function = lambda x: np.dot(x.values, model_weights)

    model = ArrowModel(pandas_predict_function).get_as_prediction_provider(background[0])
    prediction = simple_prediction(input_features=toExplain[0].features, outputs=model.predictAsync(toExplain).get()[0].outputs)
    shap_explainer = SHAPExplainer(background=background)
    explanation = shap_explainer.explain(prediction, model)

    answers = [-.152, -.114, 0.00304, .0525, -.0725]
    for saliency in explanation.getSaliencies():
        for i, feature_importance in enumerate(saliency.getPerFeatureImportance()):
            assert answers[i]-1e-3 <= feature_importance.getScore() <= answers[i]+1e-3


@pytest.mark.skip(reason="https://issues.redhat.com/browse/FAI-785")
def test_shap_plots():
    data = pd.DataFrame(np.random.rand(101, 5))
    background = [PredictionInput([feature(name=str(k), value=value, dtype="number") for k, value in row.items()]) for idx, row in data.iloc[:100].iterrows()]
    toExplain  = [PredictionInput([feature(name=str(k), value=value, dtype="number") for k, value in row.items()]) for idx, row in data.iloc[100:].iterrows()]

    model_weights = np.random.rand(5)
    pandas_predict_function = lambda x: np.stack([np.dot(x.values, model_weights), 2*np.dot(x.values, model_weights)], -1)

    model = ArrowModel(pandas_predict_function).get_as_prediction_provider(background[0])
    prediction = simple_prediction(input_features=toExplain[0].features, outputs=model.predictAsync(toExplain).get()[0].outputs)
    shap_explainer = SHAPExplainer(background=background)
    explanation = shap_explainer.explain(prediction, model)

    explanation.as_dataframe()
    explanation.candlestick_plot()
