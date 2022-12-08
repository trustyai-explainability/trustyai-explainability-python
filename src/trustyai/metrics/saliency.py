# pylint: disable = import-error
"""Saliency evaluation metrics"""
from typing import Union

from org.apache.commons.lang3.tuple import (
    Pair as _Pair,
)

from org.kie.trustyai.explainability.model import (
    PredictionInput,
    PredictionInputsDataDistribution,
)
from org.kie.trustyai.explainability.local import LocalExplainer

from jpype import JObject

from trustyai.model import simple_prediction, PredictionProvider
from trustyai.explainers import SHAPExplainer, LimeExplainer

from . import ExplainabilityMetrics


def impact_score(
    model: PredictionProvider,
    pred_input: PredictionInput,
    explainer: Union[LimeExplainer, SHAPExplainer],
    k: int,
    is_model_callable: bool = False,
):
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


def mean_impact_score(
    explainer: Union[LimeExplainer, SHAPExplainer],
    model: PredictionProvider,
    data: list,
    is_model_callable=False,
    k=2,
):
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
        m_is += impact_score(
            model, features, explainer, k, is_model_callable=is_model_callable
        )
    return m_is / len(data)


def classification_fidelity(
    explainer: Union[LimeExplainer, SHAPExplainer],
    model: PredictionProvider,
    inputs: list,
    is_model_callable: bool = False,
):
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


# pylint: disable = too-many-arguments
def local_saliency_f1(
    output_name: str,
    model: PredictionProvider,
    explainer: Union[LimeExplainer, SHAPExplainer],
    distribution: PredictionInputsDataDistribution,
    k: int,
    chunk_size: int,
):
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
    if not isinstance(explainer, LocalExplainer):
        # pylint: disable = protected-access
        local_explainer = JObject(explainer._explainer, LocalExplainer)
    else:
        local_explainer = explainer
    return ExplainabilityMetrics.getLocalSaliencyF1(
        output_name, model, local_explainer, distribution, k, chunk_size
    )
