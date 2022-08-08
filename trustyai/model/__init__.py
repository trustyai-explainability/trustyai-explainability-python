# pylint: disable = import-error, too-few-public-methods, invalid-name, duplicate-code,
# pylint: disable = unused-import, wrong-import-order
"""General model classes"""
import uuid as _uuid
from typing import List, Optional, Union, Callable
import pandas as pd
import pyarrow as pa
import numpy as np

from trustyai import _default_initializer
from trustyai.model.domain import feature_domain
from trustyai.utils import JImplementsWithDocstring

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
    Dataset as _Dataset,
)

from org.apache.arrow.vector import VectorSchemaRoot as _VectorSchemaRoot
from org.trustyai.arrowconverters import ArrowConverters, PPAWrapper
from org.kie.kogito.explainability.model.domain import (
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

trusty_type_map = {"i": "number", "O": "categorical", "f": "number", "b": "bool"}


# pylint: disable = no-member
@_jcustomizer.JImplementationFor("org.kie.kogito.explainability.model.Dataset")
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
        prediction_inputs = Dataset.df_to_prediction_object(
            df.loc[:, ~df.columns.isin(outputs)], feature
        )
        prediction_outputs = Dataset.df_to_prediction_object(df[outputs], output)
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
        prediction_inputs = Dataset.numpy_to_prediction_object(
            array[:, inputs], feature
        )
        prediction_outputs = Dataset.numpy_to_prediction_object(
            array[:, outputs], output
        )
        predictions = [
            SimplePrediction(prediction_inputs[i], prediction_outputs[i])
            for i in range(len(prediction_inputs))
        ]
        return _Dataset(predictions)

    # pylint: disable=comparison-with-callable
    @staticmethod
    def df_to_prediction_object(
        df: pd.DataFrame, func
    ) -> Union[List[PredictionInput], List[PredictionOutput]]:
        """
        Convert a Pandas DataFrame into a list of TrustyAI
        :class:`PredictionInput` or :class:`PredictionOutput`

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            The DataFrame to be converted.
        func : :class:`~feature` or :class:`~output`
            The function to use when converted the DataFrame. If ``feature``, the DataFrame
            will be wrapped into a :class:`PredictionInput`. If ``output``, the DataFrame
            will be wrapped into a :class:`PredictionOutput`.
        """
        df = df.reset_index(drop=True)
        features_names = [str(x) for x in df.columns.values]
        rows = df.values.tolist()
        types = [trusty_type_map[t.kind] for t in df.dtypes.values]
        typed_rows = [zip(row, types, features_names) for row in rows]
        predictions = []
        wrapper = PredictionInput if func is feature else PredictionOutput

        for row in typed_rows:
            values = list(row)
            collection = []
            for fv in values:
                f = func(name=fv[2], dtype=fv[1], value=fv[0])
                collection.append(f)
            predictions.append(wrapper(collection))
        return predictions

    @staticmethod
    def numpy_to_prediction_object(
        array: np.ndarray, func
    ) -> Union[List[PredictionInput], List[PredictionOutput]]:
        """
        Convert a Numpy array into a list of TrustyAI
        :class:`PredictionInput` or :class:`PredictionOutput`

        Parameters
        ----------
        array : :class:`np.ndarray`
            The array to be converted.
        func: :class:`~feature` or :class:`~output`
            The function to use when converted the array. If ``feature``, the array
            will be wrapped into a :class:`PredictionInput`. If ``output``, the array
            will be wrapped into a :class:`PredictionOutput`.
        """
        shape = array.shape

        # pylint: disable=comparison-with-callable
        if func == feature:
            prefix = "input"
            wrapper = PredictionInput
        else:
            prefix = "output"
            wrapper = PredictionOutput
        names = [f"{prefix}-{i}" for i in range(shape[1])]
        types = [trusty_type_map[array[:, i].dtype.kind] for i in range(shape[1])]
        predictions = []
        for row_index in range(shape[0]):
            collection = []
            for col_index in range(shape[1]):
                f = func(
                    name=names[col_index],
                    dtype=types[col_index],
                    value=array[row_index, col_index],
                )
                collection.append(f)
            predictions.append(wrapper(collection))
        return predictions

    @staticmethod
    def prediction_object_to_numpy(
        objects: Union[List[PredictionInput], List[PredictionOutput]]
    ) -> np.array:
        """
        Convert a list of TrustyAI
        :class:`PredictionInput` or :class:`PredictionOutput` into a Numpy array.

        Parameters
        ----------
        objects : List[:class:`PredictionInput`] orList[:class:`PredictionOutput`]
            The PredictionInput or PredictionOutput objects to be converted.
        """
        if isinstance(objects[0], PredictionInput):
            arr = np.array(
                [
                    [f.getValue().getUnderlyingObject() for f in pi.getFeatures()]
                    for pi in objects
                ]
            )
        else:
            arr = np.array(
                [
                    [o.getValue().getUnderlyingObject() for o in po.getOutputs()]
                    for po in objects
                ]
            )
        return arr

    @staticmethod
    def prediction_object_to_pandas(
        objects: Union[List[PredictionInput], List[PredictionOutput]]
    ) -> pd.DataFrame:
        """
        Convert a list of TrustyAI
        :class:`PredictionInput` or :class:`PredictionOutput` into a Pandas DataFrame.

        Parameters
        ----------
        objects : List[:class:`PredictionInput`] orList[:class:`PredictionOutput`]
            The PredictionInput or PredictionOutput objects to be converted.
        """
        if isinstance(objects[0], PredictionInput):
            df = pd.DataFrame(
                [
                    {
                        in_feature.getName(): in_feature.getValue().getUnderlyingObject()
                        for in_feature in pi.getFeatures()
                    }
                    for pi in objects
                ]
            )
        else:
            df = pd.DataFrame(
                [
                    {
                        output.getName(): output.getValue().getUnderlyingObject()
                        for output in po.getOutputs()
                    }
                    for po in objects
                ]
            )
        return df


@JImplementsWithDocstring(
    "org.kie.kogito.explainability.model.PredictionProvider", deferred=False
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
    "org.trustyai.arrowconverters.PredictionProviderArrow", deferred=False
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
    "org.kie.kogito.explainability.model.PredictionProvider", deferred=False
)
class Model:
    """Model(predict_fun, pandas=False, arrow=False)

    Wrap any Python predictive model. TrustyAI uses the :class:`Model` class to allow any Python
    predictive model to interface with the TrustyAI Java library.
    """

    def __init__(self, predict_fun, dataframe=False, output_names=None, arrow=False):
        """
        Wrap the model as a TrustyAI :obj:`PredictionProvider` Java class.

        Parameters
        ----------
        predict_fun : Callable[:class:`pandas.DataFrame`] or Callable[:class:`numpy.array`]
            A function that takes in a Numpy array or Pandas DataFrame as input and outputs a
            Pandas DataFrame or Numpy array. In general, the ``model.predict`` functions of
            sklearn-style models meet this requirement.
        dataframe: bool
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
            if not dataframe:
                self.prediction_provider_arrow = PredictionProviderArrow(
                    lambda x: predict_fun(x.values)
                )
            else:
                self.prediction_provider_arrow = PredictionProviderArrow(predict_fun)
        else:
            self.prediction_provider_arrow = None
            if dataframe:
                self.prediction_provider = PredictionProvider(
                    lambda x: self._cast_outputs(
                        predict_fun(Dataset.prediction_object_to_pandas(x))
                    )
                )
            else:
                self.prediction_provider = PredictionProvider(
                    lambda x: self._cast_outputs(
                        predict_fun(Dataset.prediction_object_to_numpy(x))
                    )
                )

    def _cast_outputs(self, output_array):
        if isinstance(output_array, np.ndarray):
            dataframe = pd.DataFrame(output_array, columns=self.output_names)
            objs = Dataset.df_to_prediction_object(dataframe, output)
        elif isinstance(output_array, pd.DataFrame):
            objs = Dataset.df_to_prediction_object(output_array, output)
        else:
            raise ValueError(
                "Unsupported output type: {}, must be numpy.ndarray or pandas.DataFrame".format(
                    type(output_array)
                )
            )
        return objs

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

        return self.prediction_provider.predictAsync(inputs)

    def __call__(self, inputs):
        """
        Alias of ``model.predict_fun(inputs)``.

        Parameters
        ----------
        inputs : Inputs to pass to the model's original `predict_fun`
        """
        return self.predict_fun(inputs)


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


# pylint: disable=line-too-long
def simple_prediction(
    input_features: Union[np.ndarray, pd.DataFrame, List[Feature], PredictionInput],
    outputs: Union[np.ndarray, pd.DataFrame, List[Output], PredictionOutput],
) -> SimplePrediction:
    """Wrap features and outputs into a SimplePrediction. Given a list of features and outputs,
    this function will bundle them into Prediction objects for use with the LIME and SHAP
    explainers.

    Parameters
    ----------
    input_features : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Feature`], or :class:`PredictionInput`
        The input features to the model, as a:

        * Numpy array of shape ``[1, n_features]``
        * Pandas DataFrame with 1 row and ``n_features`` columns
        * A List of TrustyAI :class:`Feature`, as created by the :func:`~feature` function
        * A TrustyAI :class:`PredictionInput`

    outputs : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Output`], or :class:`PredictionOutput`
        The corresponding model outputs for the provided features, that is,
        ``outputs = model(input_features)``. These can take the form of a:

        * Numpy array of shape ``[1, n_outputs]``
        * Pandas DataFrame with 1 row and ``n_outputs`` columns
        * A List of TrustyAI :class:`Output`, as created by the :func:`~output` function
        * A TrustyAI :class:`PredictionOutput`

    """
    # map inputs to PredictionInput
    if isinstance(input_features, np.ndarray):
        if len(input_features.shape) == 1:
            input_features = input_features.reshape(1, -1)
        inputs_ta = Dataset.numpy_to_prediction_object(input_features, feature)[0]
    elif isinstance(input_features, pd.DataFrame):
        inputs_ta = Dataset.df_to_prediction_object(input_features, feature)[0]
    elif isinstance(input_features, PredictionInput):
        inputs_ta = input_features
    else:
        inputs_ta = PredictionInput(input_features)

    # map outputs to PredictionOutput
    if isinstance(outputs, np.ndarray):
        if len(outputs.shape) == 1:
            outputs = outputs.reshape(1, -1)
        outputs_ta = Dataset.numpy_to_prediction_object(outputs, output)[0]
    elif isinstance(outputs, pd.DataFrame):
        outputs_ta = Dataset.df_to_prediction_object(outputs, output)[0]
    elif isinstance(outputs, PredictionOutput):
        outputs_ta = outputs
    else:
        outputs_ta = PredictionOutput(outputs)
    return SimplePrediction(inputs_ta, outputs_ta)


# pylint: disable=too-many-arguments
def counterfactual_prediction(
    input_features: Union[np.ndarray, pd.DataFrame, List[Feature], PredictionInput],
    outputs: Union[np.ndarray, pd.DataFrame, List[Output], PredictionOutput],
    data_distribution: Optional[DataDistribution] = None,
    uuid: Optional[_uuid.UUID] = None,
    timeout: Optional[float] = None,
) -> CounterfactualPrediction:
    """Wrap features and outputs into a CounterfactualPrediction. Given a list of features and
    outputs, this function will bundle them into Prediction objects for use with the
    :class:`CounterfactualExplainer`.

    Parameters
    ----------
    input_features : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Feature`], or :class:`PredictionInput`
        List of input features, as a:

        * Numpy array of shape ``[1, n_features]``
        * Pandas DataFrame with 1 row and ``n_features`` columns
        * A List of TrustyAI :class:`Feature`, as created by the :func:`~feature` function
        * A TrustyAI :class:`PredictionInput`

    outputs : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Output`], or :class:`PredictionOutput`
        The desired model outputs to be searched for in the counterfactual explanation.
        These can take the form of a:

        * Numpy array of shape ``[1, n_outputs]``
        * Pandas DataFrame with 1 row and ``n_outputs`` columns
        * A List of TrustyAI :class:`Output`, as created by the :func:`~output` function
        * A TrustyAI :class:`PredictionOutput`

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

    # map inputs to PredictionInput
    if isinstance(input_features, np.ndarray):
        if len(input_features.shape) == 1:
            input_features = input_features.reshape(1, -1)
        inputs_ta = Dataset.numpy_to_prediction_object(input_features, feature)[0]
    elif isinstance(input_features, pd.DataFrame):
        inputs_ta = Dataset.df_to_prediction_object(input_features, feature)[0]
    elif isinstance(input_features, PredictionInput):
        inputs_ta = input_features
    else:
        inputs_ta = PredictionInput(input_features)

    # map outputs to PredictionOutput
    if isinstance(outputs, np.ndarray):
        if len(outputs.shape) == 1:
            outputs = outputs.reshape(1, -1)
        outputs_ta = Dataset.numpy_to_prediction_object(outputs, output)[0]
    elif isinstance(outputs, pd.DataFrame):
        outputs_ta = Dataset.df_to_prediction_object(outputs, output)[0]
    elif isinstance(outputs, PredictionOutput):
        outputs_ta = outputs
    else:
        outputs_ta = PredictionOutput(outputs)

    return CounterfactualPrediction(
        inputs_ta,
        outputs_ta,
        data_distribution,
        uuid,
        timeout,
    )
