# pylint: disable = import-error
"""General model classes"""
from org.kie.kogito.explainability.model import (
    PerturbationContext as _PerturbationContext,
    Feature as _Feature,
    FeatureFactory as _FeatureFactory,
    PredictionInput as _PredictionInput,
    Prediction as _Prediction,
)

PerturbationContext = _PerturbationContext
Feature = _Feature
FeatureFactory = _FeatureFactory
PredictionInput = _PredictionInput
Prediction = _Prediction
