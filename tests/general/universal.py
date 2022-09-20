# General Setup
from trustyai.model import Model, simple_prediction, counterfactual_prediction
from trustyai.explainers import *


import numpy as np
import pandas as pd
import pytest

np.random.seed(0)

@pytest.mark.skip("redundant")
def test_all_explainers():
    # universal setup ==============================================================================
    data = pd.DataFrame(np.random.rand(1, 5))
    model_weights = np.random.rand(5)
    predict_function = lambda x: np.dot(x.values, model_weights)
    model = Model(predict_function, dataframe_input=True, arrow=True)
    prediction = simple_prediction(input_features=data, outputs=model(data))

    # SHAP =========================================================================================
    background = pd.DataFrame(np.zeros([100, 5]))
    shap_explainer = SHAPExplainer(background=background)
    explanation = shap_explainer.explain(prediction, model)

    for score in explanation.as_dataframe()['SHAP Value'].iloc[1:-1]:
        assert score > 0

    # LIME =========================================================================================
    explainer = LimeExplainer(samples=100, perturbations=2, seed=23, normalise_weights=False)
    explanation = explainer.explain(prediction, model)
    for score in explanation.as_dataframe()["output-0_score"]:
        assert score > 0

    # Counterfactual ===============================================================================
    features = [feature(str(k), "number", v, domain=(-10., 10.)) for k, v in data.iloc[0].items()]
    goal = np.array([[0]])
    cf_prediction = counterfactual_prediction(input_features=features, outputs=goal)
    explainer = CounterfactualExplainer(steps=10_000)
    explanation = explainer.explain(cf_prediction, model)
    result_output = model(explanation.get_proposed_features_as_pandas())
    assert result_output < .01
    assert result_output > -.01
