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
from org.kie.kogito.explainability.local.counterfactual.entities import (
    CounterfactualEntity,
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
# pylint: disable=no-member
class _JOutput:
    """Java Output implicit methods"""

    @property
    def name(self) -> str:
        """Get output's name"""
        return self.getName()

    @property
    def score(self) -> float:
        """Get output's score"""
        return self.getScore()

    @property
    def type(self):
        """Get output's type"""
        return self.getType()

    @property
    def value(self):
        """Get output's value"""
        return self.getValue()

    def __str__(self):
        return self.toString()

    def __repr__(self):
        return self.__str__()


@_jcustomizer.JImplementationFor("org.kie.kogito.explainability.model.PredictionOutput")
# pylint: disable=no-member
class _JPredictionOutput:
    """Java PredictionOutput implicit methods"""

    @property
    def outputs(self):
        """Get outputs"""
        return self.getOutputs()

    def by_name(self, name: str):
        """Get output by name"""
        return self.getByName(name)


# implicit conversion
@_jcustomizer.JImplementationFor(
    "org.kie.kogito.explainability.local.counterfactual.CounterfactualResult"
)
# pylint: disable=no-member
class _JCounterfactualResult:
    """Java CounterfactualResult implicit methods"""

    @property
    def entities(self) -> List[CounterfactualEntity]:
        """Return entities"""
        return self.getEntities()

    @property
    def output(self):
        """Return PredictionOutput"""
        return self.getOutput()


@_jcustomizer.JImplementationFor(
    "org.kie.kogito.explainability.local.counterfactual.entities.CounterfactualEntity"
)
# pylint: disable=no-member, too-few-public-methods
class _JCounterfactualEntity:
    """Java DoubleEntity implicit methods"""

    def as_feature(self) -> Feature:
        """Return as feature"""
        return self.asFeature()


@_jcustomizer.JImplementationFor("org.kie.kogito.explainability.model.Feature")
# pylint: disable=no-member
class _JFeature:
    """Java Feature implicit methods"""

    @property
    def name(self):
        """Return name"""
        return self.getName()

    @property
    def type(self):
        """Return type"""
        return self.getType()

    @property
    def value(self):
        """Return value"""
        return self.getValue()

    def __str__(self):
        return self.toString()


@_jcustomizer.JImplementationFor("org.kie.kogito.explainability.model.Value")
# pylint: disable=no-member
class _JValue:
    """Java Value implicit methods"""

    def as_string(self) -> str:
        """Return as string"""
        return self.asString()

    def as_number(self) -> float:
        """Return as number"""
        return self.asNumber()

    def __str__(self):
        return self.toString()


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
