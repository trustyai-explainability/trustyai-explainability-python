from org.apache.commons.lang3.tuple import (
    Pair as _Pair,
)


from trustyai.model import simple_prediction
from trustyai.explainers.lime import LimeExplainer
from trustyai.explainers.shap import SHAPExplainer
from trustyai.explainers import LocalExplainer

from . import ExplainabilityMetrics


def impact_score(model, pred_input, explainer, k, is_model_callable=False):
    if is_model_callable:
        output = model(pred_input)
    else:
        output = model.predict([pred_input])[0].outputs
    pred = simple_prediction(pred_input, output)
    explanation = explainer.explain(inputs=pred_input, outputs=output, model=model)
    saliency = list(explanation.saliency_map().values())[0]
    top_k_features = saliency.getTopFeatures(k)
    return ExplainabilityMetrics.impactScore(model, pred, top_k_features)


def mean_impact_score(explainer, model, data, is_model_callable=False, k=2):
    m_is = 0
    for features in data:
        m_is += impact_score(model, features, explainer, k, is_model_callable=is_model_callable)
    return m_is/len(data)


def classification_fidelity(explainer, model, inputs, is_model_callable=False):
    pairs = []
    for c_input in inputs:
        if is_model_callable:
            output = model(c_input)
        else:
            output = model.predict([c_input])[0].outputs
        explanation = explainer.explain(inputs=c_input, outputs=output, model=model)
        saliency = list(explanation.saliency_map().values())[0]
        pairs.append(_Pair.of(saliency, simple_prediction(c_input, output)))
    return ExplainabilityMetrics.classificationFidelity(pairs)


def local_saliency_f1(output_name, model, explainer, distribution, k, chunk_size):
    local_explainer = None
    if isinstance(explainer, LimeExplainer):
        local_explainer = LocalExplainer(explainer._explainer)
    elif isinstance(explainer, SHAPExplainer):
        local_explainer = LocalExplainer(explainer._explainer)
    else:
        raise ValueError("wrong explaienr type")

    return ExplainabilityMetrics.getLocalSaliencyF1(output_name, model, local_explainer, distribution, k, chunk_size)

