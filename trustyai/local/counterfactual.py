# pylint: disable=import-error
"""Counterfactual helper methods"""
from typing import List, Optional
import uuid as _uuid
from java.lang import Long
from org.kie.kogito.explainability.model import (
    CounterfactualPrediction,
    DataDistribution,
    PredictionFeatureDomain,
    PredictionInput,
    PredictionOutput,
)


# pylint: disable=too-many-arguments
def counterfactual_prediction(
    input_: PredictionInput,
    output: PredictionOutput,
    constraints: List[bool],
    domain: PredictionFeatureDomain,
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
        input_, output, domain, constraints, data_distribution, uuid, timeout
    )
