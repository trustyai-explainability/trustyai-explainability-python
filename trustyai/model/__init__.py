# pylint: disable = import-error, too-few-public-methods, invalid-name, duplicate-code
"""General model classes"""
import uuid as _uuid
from typing import List, Optional
import pandas as pd
import pyarrow as pa
import numpy as np

from java.lang import Long
from java.util.concurrent import CompletableFuture
from jpype import JImplements, JOverride, _jcustomizer, _jclass, JByte, JArray, JLong
from org.kie.kogito.explainability.local.counterfactual.entities import (
    CounterfactualEntity,
)
from org.kie.kogito.explainability.model import (
    CounterfactualPrediction as _CounterfactualPrediction,
    DataDistribution,
    DataDomain as _DataDomain,
    Feature,
    FeatureFactory as _FeatureFactory,
    Output as _Output,
    PredictionFeatureDomain as _PredictionFeatureDomain,
    PredictionInput as _PredictionInput,
    PredictionOutput as _PredictionOutput,
    Prediction as _Prediction,
    Saliency as _Saliency,
    SimplePrediction as _SimplePrediction,
    Value as _Value,
    Type as _Type,
)

from org.apache.arrow.vector import VectorSchemaRoot as _VectorSchemaRoot
from org.trustyai.arrowconverters import ArrowConverters, PPAWrapper
from org.kie.kogito.explainability.model.domain import (
    EmptyFeatureDomain as _EmptyFeatureDomain,
)

from trustyai.model.domain import feature_domain

CounterfactualPrediction = _CounterfactualPrediction
DataDomain = _DataDomain
FeatureFactory = _FeatureFactory
Output = _Output
PredictionFeatureDomain = _PredictionFeatureDomain
Prediction = _Prediction
PredictionInput = _PredictionInput
PredictionOutput = _PredictionOutput
Saliency = _Saliency
SimplePrediction = _SimplePrediction
VectorSchemaRoot = _VectorSchemaRoot
Value = _Value
Type = _Type


@JImplements("org.kie.kogito.explainability.model.PredictionProvider", deferred=False)
class Model:
    """Python transformer for the TrustyAI Java PredictionProvider"""

    def __init__(self, predict_fun):
        self.predict_fun = predict_fun

    @JOverride
    def predictAsync(self, inputs: List[PredictionInput]) -> CompletableFuture:
        """Python implementation of the predictAsync interface method"""
        future = CompletableFuture.completedFuture(
            _jclass.JClass("java.util.Arrays").asList(self.predict_fun(inputs))
        )
        return future


@JImplements("org.trustyai.arrowconverters.PredictionProviderArrow", deferred=False)
class ArrowModel:
    """Python transformer for the TrustyAI Java PredictionProviderArrow
    The argument pandas_predict_function needs to accept a pandas dataframe of
        shape (n_rows x n_features)
    and return a numpy array/dataframe of shape (n_rows x n_outputs)
    """

    def __init__(self, pandas_predict_function):
        self.pandas_predict_function = pandas_predict_function

    def predict(self, inbound_bytearray):
        """convert some inbound bytearray into dataframe, call predict function,
        then wrap back into byte array"""
        with pa.ipc.open_file(inbound_bytearray) as reader:
            batch = reader.get_batch(0)
        arr = batch.to_pandas()
        outputs = self.pandas_predict_function(arr)
        if isinstance(outputs, np.ndarray):
            outputs = pd.DataFrame(data=outputs)
        record_batch = pa.RecordBatch.from_pandas(outputs)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_file(sink, record_batch.schema) as writer:
            writer.write_batch(record_batch)
        buffer = sink.getvalue()
        return JArray(JByte)(buffer)

    @JOverride
    def predictAsync(self, inbound_bytearray: JArray(JLong)) -> CompletableFuture:
        """Python implementation of the predictAsync interface method"""
        return CompletableFuture.completedFuture(self.predict(inbound_bytearray))

    def get_as_prediction_provider(self, prototype_prediction_input):
        """Wrap the PredictionProviderArrow into a normal TrustyAI Prediction Provider"""
        return PPAWrapper(self, prototype_prediction_input)


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


@_jcustomizer.JImplementationFor("org.kie.kogito.explainability.model.PredictionInput")
# pylint: disable=no-member
class _JPredictionInput:
    """Java PredictionInput implicit methods"""

    @property
    def features(self):
        """Get features"""
        return self.getFeatures()


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

    @property
    def domain(self):
        """Return domain"""
        _domain = self.getDomain()
        if isinstance(_domain, _EmptyFeatureDomain):
            return None
        return _domain

    @property
    def is_constrained(self):
        """Return contraint"""
        return self.isConstrained()


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

    def as_obj(self):
        """Return as object"""
        return self.getUnderlyingObject()

    def __str__(self):
        return self.toString()


@_jcustomizer.JImplementationFor(
    "org.kie.kogito.explainability.model.PredictionProvider"
)
# pylint: disable=no-member, too-few-public-methods
class _JPredictionProvider:
    """Java PredictionProvider implicit methods"""

    def predict(self, inputs: List[List[Feature]]) -> List[PredictionOutput]:
        """Return model's prediction, removing async"""
        _inputs = [PredictionInput(features) for features in inputs]
        return self.predictAsync(_inputs).get()


@_jcustomizer.JImplementationFor(
    "org.kie.kogito.explainability.model.CounterfactualPrediction"
)
# pylint: disable=no-member, too-few-public-methods
class _JCounterfactualPrediction:
    """Java CounterfactualPrediction implicit methods"""

    @property
    def domain(self):
        """Return domain"""
        return self.getDomain()

    @property
    def input(self) -> PredictionInput:
        """Return input"""
        return self.getInput()

    @property
    def output(self) -> PredictionOutput:
        """Return input"""
        return self.getOutput()

    @property
    def data_distribution(self):
        """Return data distribution"""
        return self.getDataDistribution()

    @property
    def max_running_time_seconds(self):
        """Return max running time seconds"""
        return self.getMaxRunningTimeSeconds()


@_jcustomizer.JImplementationFor(
    "org.kie.kogito.explainability.model.PredictionFeatureDomain"
)
# pylint: disable=no-member
class _JPredictionFeatureDomain:
    """Java PredictionFeatureDomain implicit methods"""

    @property
    def feature_domains(self):
        """Return feature domains"""
        return self.getFeatureDomains()


def output(name, dtype, value=None, score=1.0) -> _Output:
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


def feature(name: str, dtype: str, value=None, domain=None) -> Feature:
    """Helper method to build features"""

    if dtype == "categorical":
        _factory = FeatureFactory.newCategoricalFeature
    elif dtype == "number":
        _factory = FeatureFactory.newNumericalFeature
    elif dtype == "bool":
        _factory = FeatureFactory.newBooleanFeature
    else:
        _factory = FeatureFactory.newObjectFeature

    if domain:
        _feature = _factory(name, value, feature_domain(domain))
    else:
        _feature = _factory(name, value)
    return _feature


def simple_prediction(
    input_features: List[Feature],
    outputs: List[Output],
) -> SimplePrediction:
    """Helper to build SimplePrediction"""
    return SimplePrediction(PredictionInput(input_features), PredictionOutput(outputs))


# pylint: disable=too-many-arguments
def counterfactual_prediction(
    input_features: List[Feature],
    outputs: List[Output],
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
        data_distribution,
        uuid,
        timeout,
    )
