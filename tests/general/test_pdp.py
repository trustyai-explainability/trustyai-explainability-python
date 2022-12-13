# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""PDP test suite"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from trustyai.explainers import PDPExplainer
from trustyai.model import Model
from trustyai.utils import TestModels


def create_random_df():
    X, _ = make_classification(n_samples=5000, n_features=5, n_classes=2,
                               n_clusters_per_class=2, class_sep=2, flip_y=0, random_state=23)

    return pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'x3': X[:, 2],
        'x4': X[:, 3],
        'x5': X[:, 4],
    })


def test_pdp_sumskip():
    """Test PDP with sum skip model on random generated data"""

    df = create_random_df()
    model = TestModels.getSumSkipModel(0)
    pdp_explainer = PDPExplainer()
    pdp_results = pdp_explainer.explain(model, df)
    assert pdp_results is not None
    assert pdp_results.as_dataframe() is not None


def test_pdp_sumthreshold():
    """Test PDP with sum threshold model on random generated data"""

    df = create_random_df()
    model = TestModels.getLinearThresholdModel([0.1, 0.2, 0.3, 0.4, 0.5], 0)
    pdp_explainer = PDPExplainer()
    pdp_results = pdp_explainer.explain(model, df)
    assert pdp_results is not None
    assert pdp_results.as_dataframe() is not None


def pdp_plots(block):
    """Test PDP plots"""
    np.random.seed(0)
    data = pd.DataFrame(np.random.rand(101, 5))

    model_weights = np.random.rand(5)
    predict_function = lambda x: np.stack([np.dot(x.values, model_weights), 2 * np.dot(x.values, model_weights)], -1)
    model = Model(predict_function, dataframe_input=True)
    pdp_explainer = PDPExplainer()
    explanation = pdp_explainer.explain(model, data)

    explanation.plot(block=block)
    explanation.plot(block=block, output_name='output-0')


@pytest.mark.block_plots
def test_lime_plots_blocking():
    pdp_plots(True)


def test_lime_plots():
    pdp_plots(False)
