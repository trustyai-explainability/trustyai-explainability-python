# pylint: disable=import-error, wrong-import-position, wrong-import-order, duplicate-code, unused-import
"""SHAP explainer test suite"""

from common import *

import pandas as pd
import numpy as np

np.random.seed(0)

import pytest
from trustyai.explainers import SHAPExplainer
from trustyai.model import feature, Model
from trustyai.utils.data_conversions import numpy_to_prediction_object
from trustyai.utils import TestModels


def test_no_variance_one_output():
    """Check if the explanation returned is not null"""
    model = TestModels.getSumSkipModel(0)

    background = np.array([[1.0, 2.0, 3.0] for _ in range(2)])
    prediction_outputs = model.predictAsync(numpy_to_prediction_object(background, feature)).get()
    shap_explainer = SHAPExplainer(background=background)
    for i in range(2):
        explanation = shap_explainer.explain(inputs=background[i], outputs=prediction_outputs[i].outputs, model=model)
        for _, saliency in explanation.saliency_map().items():
            for feature_importance in saliency.getPerFeatureImportance()[:-1]:
                assert feature_importance.getScore() == 0.0


def test_shap_arrow():
    """Basic SHAP/Arrow test"""
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(101, 5))
    background = data.iloc[:100]
    to_explain = data.iloc[100:101]

    model_weights = np.random.rand(5)
    predict_function = lambda x: np.dot(x.values, model_weights)

    model = Model(predict_function, dataframe_input=True)
    shap_explainer = SHAPExplainer(background=background)
    explanation = shap_explainer.explain(inputs=to_explain, outputs=model(to_explain), model=model)


    answers = [-.152, -.114, 0.00304, .0525, -.0725]
    for _, saliency in explanation.saliency_map().items():
        for i, feature_importance in enumerate(saliency.getPerFeatureImportance()[:-1]):
            assert answers[i] - 1e-2 <= feature_importance.getScore() <= answers[i] + 1e-2


def shap_plots(block):
    """Test SHAP plots"""
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(101, 5))
    background = data.iloc[:100]
    to_explain = data.iloc[100:101]

    model_weights = np.random.rand(5)
    predict_function = lambda x: np.stack([np.dot(x.values, model_weights), 2 * np.dot(x.values, model_weights)], -1)
    model = Model(predict_function, dataframe_input=True)
    shap_explainer = SHAPExplainer(background=background)
    explanation = shap_explainer.explain(inputs=to_explain, outputs=model(to_explain), model=model)

    explanation.plot(block=block)
    explanation.plot(block=block, render_bokeh=True)
    explanation.plot(block=block, output_name='output-0')
    explanation.plot(block=block, output_name='output-0', render_bokeh=True)


@pytest.mark.block_plots
def test_shap_plots_blocking():
    shap_plots(block=True)


def test_shap_plots():
    shap_plots(block=False)


def test_shap_as_df():
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(101, 5))
    background = data.iloc[:100].values
    to_explain = data.iloc[100:101].values

    model_weights = np.random.rand(5)
    predict_function = lambda x: np.stack([np.dot(x, model_weights), 2 * np.dot(x, model_weights)], -1)

    model = Model(predict_function, disable_arrow=True)

    shap_explainer = SHAPExplainer(background=background)
    explanation = shap_explainer.explain(inputs=to_explain, outputs=model(to_explain), model=model)

    for out_name, df in explanation.as_dataframe().items():
        assert "Mean Background Value" in df
        assert "output" in out_name
        assert all([x in str(df) for x in "01234"])


def test_shap_as_html():
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(101, 5))
    background = data.iloc[:100].values
    to_explain = data.iloc[100:101].values

    model_weights = np.random.rand(5)
    predict_function = lambda x: np.stack([np.dot(x, model_weights), 2 * np.dot(x, model_weights)], -1)

    model = Model(predict_function, disable_arrow=True)

    shap_explainer = SHAPExplainer(background=background)
    explanation = shap_explainer.explain(inputs=to_explain, outputs=model(to_explain), model=model)
    assert True


def test_shap_numpy():
    np.random.seed(0)
    data = np.random.rand(101, 5)
    model_weights = np.random.rand(5)
    predict_function = lambda x: np.stack([np.dot(x, model_weights), 2 * np.dot(x, model_weights)], -1)
    fnames = ['f{}'.format(x) for x in "abcde"]
    onames = ['o{}'.format(x) for x in "12"]
    model = Model(predict_function,
                  feature_names=fnames,
                  output_names=onames
                  )

    shap_explainer = SHAPExplainer(background=data[1:])
    explanation = shap_explainer.explain(inputs=data[0], outputs=model(data[0]), model=model)

    for oname in onames:
        assert oname in explanation.as_dataframe().keys()
        for fname in fnames:
            assert fname in explanation.as_dataframe()[oname]['Feature'].values
