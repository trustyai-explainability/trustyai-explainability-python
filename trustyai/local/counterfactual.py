# pylint: disable=import-error
"""Counterfactual helper methods"""
from typing import List, Optional
import uuid as _uuid
from java.lang import Long
from org.kie.kogito.explainability.model import (
    CounterfactualPrediction,
    DataDistribution,
    Feature,
    Output,
    PredictionFeatureDomain,
    PredictionInput,
    PredictionOutput,
)
from org.kie.kogito.explainability.model.domain import FeatureDomain


# pylint: disable=too-many-arguments
def counterfactual_prediction(
    input_features: List[Feature],
    outputs: List[Output],
    constraints: List[bool],
    domains: List[FeatureDomain],
    data_distribution: Optional[DataDistribution] = None,
    uuid: Optional[_uuid.UUID] = None,
    timeout: Optional[float] = None,
) -> CounterfactualPrediction:
    """Helper to build CounterfactualPrediction"""
    if not uuid:
        uuid = _uuid.uuid4()
    if timeout:
        timeout = Long(timeout)
    return CounterfactualPrediction(
        PredictionInput(input_features),
        PredictionOutput(outputs),
        PredictionFeatureDomain(domains),
        constraints,
        data_distribution,
        uuid,
        timeout,
    )
