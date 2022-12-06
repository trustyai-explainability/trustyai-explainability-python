# pylint: disable = import-error
"""Saliency evaluation metrics"""
from org.apache.commons.lang3.tuple import (
    Pair as _Pair,
)

from trustyai.model import simple_prediction
from trustyai.explainers.lime import LimeExplainer
from trustyai.explainers.shap import SHAPExplainer
from trustyai.explainers import LocalExplainer

from . import ExplainabilityMetrics


def impact_score(model, pred_input, explainer, k, is_model_callable=False):
    """
    Parameters
    ----------
    model: trustyai.PredictionProvider
        the model used to generate predictions
    pred_input: trustyai.PredictionInput
        the input to the model
    explainer: Union[trustyai.explainers.LimeExplainer, trustyai.explainers.SHAPExplainer]
        the explainer to evaluate
    k: int
        the number of top important features
    is_model_callable: bool
        whether to directly use model function call or use the predict method

    Returns
    -------
    :float:
        impact score metric
    """
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
    """
    Parameters
    ----------
    explainer: Union[trustyai.explainers.LimeExplainer, trustyai.explainers.SHAPExplainer]
        the explainer to evaluate
    model: trustyai.PredictionProvider
        the model used to generate predictions
    data: list[list[trustyai.model.Feature]]
        the inputs to calculate the metric for
    is_model_callable: bool
        whether to directly use model function call or use the predict method
    k: int
        the number of top important features

    Returns
    -------
    :float:
        the mean impact score metric across all inputs
    """
    m_is = 0
    for features in data:
        m_is += impact_score(model, features, explainer, k, is_model_callable=is_model_callable)
    return m_is / len(data)


def classification_fidelity(explainer, model, inputs, is_model_callable=False):
    """
    Parameters
    ----------
    explainer: Union[trustyai.explainers.LimeExplainer, trustyai.explainers.SHAPExplainer]
        the explainer to evaluate
    model: trustyai.PredictionProvider
        the model used to generate predictions
    inputs: list[list[trustyai.model.Feature]]
        the inputs to calculate the metric for
    is_model_callable: bool
        whether to directly use model function call or use the predict method

    Returns
    -------
    :float:
        the classification fidelity metric
    """
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
    """
    Parameters
    ----------
    output_name: str
        the name of the output to calculate the metric for
    model: trustyai.PredictionProvider
        the model used to generate predictions
    explainer: Union[trustyai.explainers.LIMEExplainer, trustyai.explainers.SHAPExplainer,
                trustyai.explainers.LocalExplainer]
        the explainer to evaluate
    distribution: org.kie.trustyai.explainability.model.PredictionInputsDataDistribution
        the data distribution to fetch the inputs from
    k: int
        the number of top important features
    chunk_size: int
        the chunk of inputs to fetch fro the distribution

    Returns
    -------
    :float:
        the local saliency f1 metric
    """
    if isinstance(explainer, LimeExplainer):
        local_explainer = LocalExplainer(explainer._explainer)
    elif isinstance(explainer, SHAPExplainer):
        local_explainer = LocalExplainer(explainer._explainer)
    elif isinstance(explainer, LocalExplainer):
        local_explainer = explainer
    else:
        raise ValueError(f"Wrong explainer type '{explainer}', "
                         f"expected one of [LimeExplainer, SHAPExplainer, LocalExplainer]")

    return ExplainabilityMetrics.getLocalSaliencyF1(output_name, model, local_explainer,
                                                    distribution, k, chunk_size)
