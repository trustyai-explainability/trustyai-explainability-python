import os
import sys

from trustyai.explainers import SHAPResults
from trustyai.metrics import ExplainabilityMetrics
from trustyai.model import simple_prediction

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../general/")

def saliency_impact_score_benchmark(model, input, explainer, k=2, is_model_callable=False):
    if is_model_callable:
        output = model(input)
    else:
        output = model.predict([input])[0].outputs
    pred = simple_prediction(input, output)
    explanation = explainer.explain(inputs=input, outputs=output, model=model)
    if isinstance(explanation, SHAPResults):
        saliency = list(explanation.get_saliencies().values())[0]
    else:
        saliency = list(explanation.map().values())[0]
    top_k_features = saliency.getTopFeatures(k)
    return ExplainabilityMetrics.impactScore(model, pred, top_k_features)


def mean_impact_score(explainer, model, data, is_model_callable=False):
    m_is = 0
    for features in data:
        m_is += saliency_impact_score_benchmark(model, features, explainer, is_model_callable=is_model_callable)
    return m_is