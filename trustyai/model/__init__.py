# pylint: disable = import-error, too-few-public-methods, invalid-name, duplicate-code
"""General model classes"""
from typing import List
from java.util.concurrent import CompletableFuture, ForkJoinPool
from jpype import JImplements, JOverride, JProxy, _jcustomizer
from org.kie.kogito.explainability.model import (
    CounterfactualPrediction as _CounterfactualPrediction,
    DataDomain as _DataDomain,
    PerturbationContext as _PerturbationContext,
    Feature as _Feature,
    FeatureFactory as _FeatureFactory,
    Output as _Output,
    PredictionFeatureDomain as _PredictionFeatureDomain,
    PredictionInput as _PredictionInput,
    PredictionOutput as _PredictionOutput,
    Prediction as _Prediction,
    SimplePrediction as _SimplePrediction,
    Value as _Value,
    Type as _Type,
)
from trustyai.utils import toJList

CounterfactualPrediction = _CounterfactualPrediction
DataDomain = _DataDomain
PerturbationContext = _PerturbationContext
Feature = _Feature
FeatureFactory = _FeatureFactory
Output = _Output
PredictionFeatureDomain = _PredictionFeatureDomain
Prediction = _Prediction
PredictionInput = _PredictionInput
PredictionOutput = _PredictionOutput
SimplePrediction = _SimplePrediction
Value = _Value
Type = _Type


class InnerSupplier:
    """Wraps the Python predict function in a Java Supplier"""

    def __init__(self, _inputs: List[PredictionInput], predict_fun):
        self.inputs = _inputs
        self.predict_fun = predict_fun

    def get(self) -> List[PredictionOutput]:
        """The Supplier interface get method"""
        return self.predict_fun(self.inputs)


@JImplements("org.kie.kogito.explainability.model.PredictionProvider", deferred=True)
class PredictionProvider:
    """Python transformer for the TrustyAI Java PredictionProvider"""

    def __init__(self, predict_fun):
        self.predict_fun = predict_fun

    @JOverride
    def predictAsync(self, inputs: List[PredictionInput]) -> CompletableFuture:
        """Python implementation of the predictAsync interface method"""
        supplier = InnerSupplier(inputs, self.predict_fun)
        proxy = JProxy("java.util.function.Supplier", inst=supplier)
        future = CompletableFuture.supplyAsync(proxy, ForkJoinPool.commonPool())
        return future
