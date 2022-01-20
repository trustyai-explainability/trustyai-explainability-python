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


@_jcustomizer.JImplementationFor("org.kie.kogito.explainability.model.Output")
class _JOutput:
    """Java Output implicit methods"""

    # pylint: disable=no-member
    def __str__(self):
        return (
            f"Output(name={self.getName()}, type={self.getType()}, "
            f"value={self.getValue()}, score={self.getScore()})"
        )

    def __repr__(self):
        return self.__str__()


def output(name, dtype, value=None, score=1.0):
    """Helper method returning a Java Output"""
    if dtype == "text":
        _type = Type.TEXT
    elif dtype == "number":
        _type = Type.NUMBER
    elif dtype == "bool":
        _type = Type.BOOLEAN
    elif dtype == "categorical":
        _type = Type.CATEGORICAL
    else:
        _type = Type.UNDEFINED
    return _Output(name, _type, Value(value), score)
