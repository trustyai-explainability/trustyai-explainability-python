# pylint: disable=import-error
"""Counterfactual helper methods"""
from typing import List, Optional, Tuple
import uuid as _uuid
from java.lang import Long
from jpype import _jclass
from org.kie.kogito.explainability.model import (
    CounterfactualPrediction,
    DataDistribution,
    Feature,
    Output,
    PredictionFeatureDomain,
    PredictionInput,
    PredictionOutput,
    SimplePrediction,
)
from trustyai.model.domain import feature_domain


# pylint: disable=too-many-arguments
def counterfactual_prediction(
    input_features: List[Feature],
    outputs: List[Output],
    domains: List[Optional[Tuple]],
    constraints: Optional[List[bool]] = None,
    data_distribution: Optional[DataDistribution] = None,
    uuid: Optional[_uuid.UUID] = None,
    timeout: Optional[float] = None,
) -> CounterfactualPrediction:
    """Helper to build CounterfactualPrediction"""
    if not uuid:
        uuid = _uuid.uuid4()
    if timeout:
        timeout = Long(timeout)
    if not constraints:
        constraints = [False] * len(input_features)

    # build the feature domains from the Python tuples
    java_domains = _jclass.JClass("java.util.Arrays").asList(
        [feature_domain(domain) for domain in domains]
    )

    return CounterfactualPrediction(
        PredictionInput(input_features),
        PredictionOutput(outputs),
        PredictionFeatureDomain(java_domains),
        constraints,
        data_distribution,
        uuid,
        timeout,
    )


def simple_prediction(
    input_features: List[Feature],
    outputs: List[Output],
) -> SimplePrediction:
    """Helper to build SimplePrediction"""
    return SimplePrediction(PredictionInput(input_features), PredictionOutput(outputs))
