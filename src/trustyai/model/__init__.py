# pylint: disable = import-error, too-few-public-methods, invalid-name, duplicate-code, too-many-lines
# pylint: disable = unused-import, wrong-import-order
# pylint: disable = consider-using-f-string
"""General model classes"""
import uuid as _uuid
from typing import List, Optional, Union, Callable
import pandas as pd
import pyarrow as pa
import numpy as np

from trustyai import _default_initializer
from trustyai.model.domain import feature_domain
from trustyai.utils import JImplementsWithDocstring
from trustyai.utils.data_conversions import (
    one_input_convert,
    one_output_convert,
    OneOutputUnionType,
    OneInputUnionType,
    numpy_to_prediction_object,
    df_to_prediction_object,
    prediction_object_to_numpy,
    prediction_object_to_pandas,
    data_conversion_docstring,
)

from java.lang import Long
from java.util.concurrent import CompletableFuture
from jpype import (
    JImplements,
    JOverride,
    _jcustomizer,
    _jclass,
    JByte,
    JArray,
    JLong,
    JInt,
    JString,
)
from org.kie.trustyai.explainability.local.counterfactual.entities import (
    CounterfactualEntity,
)
from org.kie.trustyai.explainability.model import (
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
    SaliencyResults as _SaliencyResults,
    SimplePrediction as _SimplePrediction,
    Value as _Value,
    Type as _Type,
    Dataset as _Dataset,
)

from org.apache.arrow.vector import VectorSchemaRoot as _VectorSchemaRoot
from org.kie.trustyai.arrow import ArrowConverters, PPAWrapper
from org.kie.trustyai.explainability.model.domain import (
    EmptyFeatureDomain as _EmptyFeatureDomain,
)

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


# pylint: disable = no-member
@_jcustomizer.JImplementationFor("org.kie.trustyai.explainability.model.Dataset")
class Dataset:
    """Wrapper class for TrustyAI Datasets."""

    @property
    def data(self):
        """Return the dataset's data"""
        return self.getData()

    @property
    def inputs(self):
        """Return the dataset's input"""
        return self.getInputs()

    @property
    def outputs(self):
        """Return the dataset's output"""
        return self.getOutputs()

    @staticmethod
    def from_df(df: pd.DataFrame, outputs: Optional[List[str]] = None) -> _Dataset:
        """Create a TrustyAI Dataset from a Pandas DataFrame.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            The Pandas DataFrame to be converted into a :class:`Dataset`.
        outputs : Optional[List[str]]
            An optional list of column names that represent model outputs. If not supplied,
            the right-most column will taken as the model output.

        Returns
        -------
        :class:`Dataset`
        """
        if not outputs:
            outputs = [df.iloc[:, -1].name]
        prediction_inputs = df_to_prediction_object(
            df.loc[:, ~df.columns.isin(outputs)], feature
        )
        prediction_outputs = df_to_prediction_object(df[outputs], output)
        predictions = [
            SimplePrediction(prediction_inputs[i], prediction_outputs[i])
            for i in range(len(prediction_inputs))
        ]
        return _Dataset(predictions)

    @staticmethod
    def from_numpy(array: np.ndarray, outputs: Optional[List[int]] = None) -> _Dataset:
        """Create a TrustyAI Dataset from a Numpy array.

        Parameters
        ----------
        array : :class:`numpy.array`
            The Numpy array to be converted into a :class:`Dataset`.
        outputs : Optional[List[int]]
            An optional list of column indeces that represent model outputs. If not supplied,
            the right-most column will taken as the model output.

        Returns
        -------
        :class:`Dataset`
        """
        shape = array.shape
        if not outputs:
            outputs = [shape[1] - 1]
        inputs = list(set(range(shape[1])).difference(outputs))
        prediction_inputs = numpy_to_prediction_object(array[:, inputs], feature)
        prediction_outputs = numpy_to_prediction_object(array[:, outputs], output)
        predictions = [
            SimplePrediction(prediction_inputs[i], prediction_outputs[i])
            for i in range(len(prediction_inputs))
        ]
        return _Dataset(predictions)


@JImplementsWithDocstring(
    "org.kie.trustyai.explainability.model.PredictionProvider", deferred=False
)
class PredictionProvider:
    """PredictionProvider(predict_fun)
    Wrapper for any predictive function.

    Wrapper for any predictive function to be explained. This implements the TrustyAI Java
    class :class:`~PredictionProvider`, which is required to interface with any
    of the explainers.
    """

    def __init__(
        self, predict_fun: Callable[[List[PredictionInput]], List[PredictionOutput]]
    ):
        """
        Create the model as a TrustyAI :obj:`PredictionProvider` Java class.

        Parameters
        ----------
        predict_fun : Callable[[List[:obj:`PredictionInput`]], List[:obj:`PredictionOutput`]]
            A function that takes a list of prediction inputs and outputs a list of prediction
            outputs.

        """
        self.predict_fun = predict_fun

    @JOverride
    def predictAsync(self, inputs: List[PredictionInput]) -> CompletableFuture:
        """
        Python implementation of the :func:`predictAsync` function with the
        TrustyAI :obj:`PredictionProvider` interface.

        Parameters
        ----------
        inputs : List[:obj:`PredictionInput`]
            A list of inputs.

        Returns
        -------
        :obj:`CompletableFuture`
            A Java :obj:`CompletableFuture` containing the model outputs.
        """
        future = CompletableFuture.completedFuture(
            _jclass.JClass("java.util.Arrays").asList(self.predict_fun(inputs))
        )
        return future


@JImplementsWithDocstring(
    "org.kie.trustyai.arrow.PredictionProviderArrow", deferred=False
)
class PredictionProviderArrow:
    """PredictionProviderArrow(pandas_predict_fun)

    Wrapper for any predictive function, optimized for Python-Java communication.

    The :class:`PredictionProviderArrow` class takes advantage of Apache Arrow to drastically
    speed up data transfer between Python and Java. We recommend using an ArrowModel
    whenever seeking LIME or SHAP explanations.
    """

    def __init__(self, predict_fun):
        """
        Create the model as a TrustyAI :obj:`PredictionProvider` Java class.

        Parameters
        ----------
        predict_fun : Callable[:class:`pd.DataFrame`, :class:`np.array`]
            A function that takes in Pandas DataFrame as input and outputs a Numpy array.
            In general, the ``model.predict`` functions of sklearn-style models meet this
            requirement.

        """
        self.predict_function = predict_fun

    def predict(self, inbound_bytearray):
        """The function called internally by :func:`predictAsync` when communicating
        between Java and Python. This function should never need to be called manually."""
        with pa.ipc.open_file(inbound_bytearray) as reader:
            batch = reader.get_batch(0)
        arr = batch.to_pandas()
        outputs = self.predict_function(arr)
        record_batch = pa.RecordBatch.from_pandas(outputs)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_file(sink, record_batch.schema) as writer:
            writer.write_batch(record_batch)
        buffer = sink.getvalue()
        return JArray(JByte)(buffer)

    @JOverride
    def predictAsync(self, inbound_bytearray: JArray(JLong)) -> CompletableFuture:
        """
        Python implementation of the :func:`predictAsync` function with the
        TrustyAI :obj:`PredictionProviderArrow` interface.

        Parameters
        ----------
        inbound_bytearray : List[:obj:`PredictionInput`]
            The inbound bytearray

        Returns
        -------
        :obj:`CompletableFuture`
            A Java :obj:`CompletableFuture` containing the model outputs.
        """
        return CompletableFuture.completedFuture(self.predict(inbound_bytearray))

    def get_as_prediction_provider(self, prototype_prediction_input):
        """
        Wrap the :class:`ArrowModel` into a TrustyAI :class:`PredictionProvider`.
        This is required to use an :class:`ArrowModel` with any of the explainers.


        Parameters
        ----------
        prototype_prediction_input : :obj:`PredictionInput`
            A single example input to the model. This is necessary to specify the data
            schema to be communicated between Python and Java.

        Returns
        -------
        :obj:`PredictionProvider`
            A TrustyAI :class:`PredictionProvider`.
        """
        return PPAWrapper(self, prototype_prediction_input)


@JImplementsWithDocstring(
    "org.kie.trustyai.explainability.model.PredictionProvider", deferred=False
)
class Model:
    """Model(predict_fun, pandas=False, arrow=False)

    Wrap any Python predictive model. TrustyAI uses the :class:`Model` class to allow any Python
    predictive model to interface with the TrustyAI Java library.
    """

    def __init__(
        self, predict_fun, dataframe_input=False, output_names=None, arrow=False
    ):
        """
        Wrap the model as a TrustyAI :obj:`PredictionProvider` Java class.

        Parameters
        ----------
        predict_fun : Callable[:class:`pandas.DataFrame`] or Callable[:class:`numpy.array`]
            A function that takes in a Numpy array or Pandas DataFrame as input and outputs a
            Pandas DataFrame or Numpy array. In general, the ``model.predict`` functions of
            sklearn-style models meet this requirement.
        dataframe_input: bool
            Whether `predict_fun` expects a :class:`pandas.DataFrame` as input.
        output_names : List[String]:
            If the model outputs a numpy array, you can specify the names of the model outputs
            here.
        arrow: bool
            Whether to use Apache arrow to speed up data transfer between Java and Python.
            In general, set this to ``true`` whenever LIME or SHAP explanations are needed,
            and ``false`` for counterfactuals.
        """
        self.arrow = arrow
        self.predict_fun = predict_fun
        self.output_names = output_names

        if arrow:
            self.prediction_provider = None
            if not dataframe_input:
                self.prediction_provider_arrow = PredictionProviderArrow(
                    lambda x: self._cast_outputs_to_dataframe(predict_fun(x.values))
                )
            else:
                self.prediction_provider_arrow = PredictionProviderArrow(
                    lambda x: self._cast_outputs_to_dataframe(predict_fun(x))
                )
        else:
            self.prediction_provider_arrow = None
            if dataframe_input:
                self.prediction_provider = PredictionProvider(
                    lambda x: self._cast_outputs(
                        predict_fun(prediction_object_to_pandas(x))
                    )
                )
            else:
                self.prediction_provider = PredictionProvider(
                    lambda x: self._cast_outputs(
                        predict_fun(prediction_object_to_numpy(x))
                    )
                )

    def _cast_outputs(self, output_array):
        return df_to_prediction_object(
            self._cast_outputs_to_dataframe(output_array), output
        )

    def _cast_outputs_to_dataframe(self, output_array):
        if isinstance(output_array, pd.DataFrame):
            out = output_array
        elif isinstance(output_array, np.ndarray):
            if self.output_names is None:
                if len(output_array.shape) == 1:
                    columns = ["output-0"]
                else:
                    columns = [
                        "output-{}".format(i) for i in range(output_array.shape[1])
                    ]
            else:
                columns = self.output_names
            out = pd.DataFrame(output_array, columns=columns)
        else:
            raise ValueError(
                "Unsupported output type: {}, must be numpy.ndarray or pandas.DataFrame".format(
                    type(output_array)
                )
            )
        return out

    @JOverride
    def predictAsync(self, inputs: List[PredictionInput]) -> CompletableFuture:
        """
        Python implementation of the :func:`predictAsync` function with the
        TrustyAI :obj:`PredictionProvider` interface.

        Parameters
        ----------
        inputs : List[:obj:`PredictionInput`]
            A list of inputs.

        Returns
        -------
        :obj:`CompletableFuture`
            A Java :obj:`CompletableFuture` containing the model outputs.
        """
        if self.arrow and self.prediction_provider is None:
            self.prediction_provider = (
                self.prediction_provider_arrow.get_as_prediction_provider(inputs[0])
            )
        out = self.prediction_provider.predictAsync(inputs)
        return out

    def __call__(self, inputs):
        """
        Alias of ``model.predict_fun(inputs)``.

        Parameters
        ----------
        inputs : Inputs to pass to the model's original `predict_fun`
        """
        return self.predict_fun(inputs)


@_jcustomizer.JImplementationFor("org.kie.trustyai.explainability.model.Output")
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


@_jcustomizer.JImplementationFor(
    "org.kie.trustyai.explainability.model.PredictionOutput"
)
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


@_jcustomizer.JImplementationFor(
    "org.kie.trustyai.explainability.model.PredictionInput"
)
# pylint: disable=no-member
class _JPredictionInput:
    """Java PredictionInput implicit methods"""

    @property
    def features(self):
        """Get features"""
        return self.getFeatures()


# implicit conversion
@_jcustomizer.JImplementationFor(
    "org.kie.trustyai.explainability.local.counterfactual.CounterfactualResult"
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
    "org.kie.trustyai.explainability.local.counterfactual.entities.CounterfactualEntity"
)
# pylint: disable=no-member, too-few-public-methods
class _JCounterfactualEntity:
    """Java DoubleEntity implicit methods"""

    def as_feature(self) -> Feature:
        """Return as feature"""
        return self.asFeature()


@_jcustomizer.JImplementationFor("org.kie.trustyai.explainability.model.Feature")
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


@_jcustomizer.JImplementationFor("org.kie.trustyai.explainability.model.Value")
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
    "org.kie.trustyai.explainability.model.PredictionProvider"
)
# pylint: disable=no-member, too-few-public-methods
class _JPredictionProvider:
    """Java PredictionProvider implicit methods"""

    def predict(self, inputs: List[List[Feature]]) -> List[PredictionOutput]:
        """Return model's prediction, removing async"""
        _inputs = [PredictionInput(features) for features in inputs]
        return self.predictAsync(_inputs).get()


@_jcustomizer.JImplementationFor(
    "org.kie.trustyai.explainability.model.CounterfactualPrediction"
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
    "org.kie.trustyai.explainability.model.PredictionFeatureDomain"
)
# pylint: disable=no-member
class _JPredictionFeatureDomain:
    """Java PredictionFeatureDomain implicit methods"""

    @property
    def feature_domains(self):
        """Return feature domains"""
        return self.getFeatureDomains()


@_jcustomizer.JImplementationFor(
    "org.kie.trustyai.explainability.model.SaliencyResults"
)
# pylint: disable=no-member
class SaliencyResults:
    """Java PredictionFeatureDomain implicit methods"""

    @property
    def saliencies(self):
        """Return saliencies"""
        return self.getSaliencies()

    def __sub__(self, other: _SaliencyResults):
        """Overload SaliencyResults difference"""
        return self.difference(other)

    def __eq__(self, other: _SaliencyResults):
        """Overload SaliencyResults equality"""
        return self.equals(other)


def output(name, dtype, value=None, score=1.0) -> _Output:
    """Create a Java :class:`Output`. The :class:`Output` class is used to represent the
    individual components of model outputs.

    Parameters
    ----------
    name : str
        The name of the given output.
    dtype: str
        The type of the given output, one of:

        * ``text`` for textual outputs.
        * ``number`` for numeric outputs.
        * ``bool`` for binary or boolean outputs.
        * ``categorical`` for categorical outputs.

        If `dtype` is unspecified or takes a different value than listed above, the
         feature type will be set as `UNDEFINED`.
    value : Any
        The value of this output.
    score : float
        The confidence of this particular output.

    Returns
    -------
    :class:`Output`
        A TrustyAI :class:`Output` object, to be used in
        the :func:`~trustyai.model.simple_prediction` or
        :func:`~trustyai.model.counterfactual_prediction` functions.

    """
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
    """Create a Java :class:`Feature`. The :class:`Feature` class is used to represent the
    individual components (or features) of input data points.

    Parameters
    ----------
    name : str
        The name of the given feature.
    dtype: str
        The type of the given feature, one of:

        * ``categorical`` for categorical features.
        * ``number`` for numeric features.
        * ``bool`` for binary or boolean features.

        If `dtype` is unspecified or takes a different value than listed above, the feature
        type will be taken as a generic object.
    value : Any
        The value of this feature.
    domain : Union[tuple, list]
        A tuple or list that defines the feature domain. A tuple will define a numeric range
        from ``[tuple[0], tuple[1])``, while a list defines the valid values for
        a categorical feature.

    Returns
    -------
    :class:`Feature`
        A TrustyAI :class:`Feature` object, to be used in the
        :func:`~trustyai.model.simple_prediction` or
        :func:`~trustyai.model.counterfactual_prediction` functions.

    """

    if dtype == "categorical":
        if isinstance(value, int):
            _factory = FeatureFactory.newCategoricalNumericalFeature
            value = JInt(value)
        else:
            _factory = FeatureFactory.newCategoricalFeature
            value = JString(value)
    elif dtype == "number":
        _factory = FeatureFactory.newNumericalFeature
    elif dtype == "bool":
        _factory = FeatureFactory.newBooleanFeature
    else:
        _factory = FeatureFactory.newObjectFeature

    name = JString(name)
    if domain:
        _feature = _factory(name, value, feature_domain(domain))
    else:
        _feature = _factory(name, value)
    return _feature


# pylint: disable=line-too-long
@data_conversion_docstring("one_input", "one_output")
def simple_prediction(
    input_features: OneInputUnionType, outputs: OneOutputUnionType
) -> SimplePrediction:
    """Wrap features and outputs into a SimplePrediction. Given a list of features and outputs,
    this function will bundle them into Prediction objects for use with the LIME and SHAP
    explainers.

    Parameters
    ----------
    input_features : {}
        List of input features, as a: {}
    outputs : {}
        The desired model outputs to be searched for in the counterfactual explanation.
        These can take the form of a: {}
    """

    return SimplePrediction(
        one_input_convert(input_features), one_output_convert(outputs)
    )


# pylint: disable=too-many-arguments
@data_conversion_docstring("one_input", "one_output")
def counterfactual_prediction(
    input_features: OneInputUnionType,
    outputs: OneOutputUnionType,
    data_distribution: Optional[DataDistribution] = None,
    uuid: Optional[_uuid.UUID] = None,
    timeout: Optional[float] = None,
) -> CounterfactualPrediction:
    """Wrap features and outputs into a CounterfactualPrediction. Given a list of features and
    outputs, this function will bundle them into Prediction objects for use with the
    :class:`CounterfactualExplainer`.

    Parameters
    ----------
    input_features : {}
        List of input features, as a: {}
    outputs : {}
        The desired model outputs to be searched for in the counterfactual explanation.
        These can take the form of a: {}
    data_distribution : Optional[:class:`DataDistribution`]
        The :class:`DataDistribution` to use when sampling the inputs.
    uuid : Optional[:class:`_uuid.UUID`]
        The UUID to use during search.
    timeout : Optional[float]
        The timeout time in seconds of the counterfactual explanation.
    """
    if not uuid:
        uuid = _uuid.uuid4()
    if timeout:
        timeout = Long(timeout)

    return CounterfactualPrediction(
        one_input_convert(input_features),
        one_output_convert(outputs),
        data_distribution,
        uuid,
        timeout,
    )
