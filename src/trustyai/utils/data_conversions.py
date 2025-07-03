"""Data Converters between Python and Java"""

# pylint: disable = import-error, line-too-long, trailing-whitespace, unused-import, cyclic-import
# pylint: disable = consider-using-f-string, invalid-name, wrong-import-order
import warnings
from typing import Union, List, Optional, Tuple
from itertools import filterfalse

import trustyai.model
from org.kie.trustyai.explainability.model import (
    Feature,
    Output,
    PredictionInput,
    PredictionOutput,
)
from org.kie.trustyai.explainability.model.dataframe import Dataframe

from org.kie.trustyai.explainability.model.domain import (
    FeatureDomain,
    EmptyFeatureDomain,
)

import pandas as pd
import numpy as np

# UNION TYPES FOR INPUTS AND OUTPUTS
# if a TrustyAI function wants AN input/output, it should accept this union type:
OneInputUnionType = Union[
    int,
    float,
    np.integer,
    np.inexact,
    np.ndarray,
    pd.DataFrame,
    pd.Series,
    List[Feature],
    PredictionInput,
]
OneOutputUnionType = Union[
    int,
    float,
    np.integer,
    np.inexact,
    np.ndarray,
    pd.DataFrame,
    pd.Series,
    List[Output],
    PredictionOutput,
]

# if a TrustyAI function wants a LIST of inputs/outputs, it should accept this union type:
ManyInputsUnionType = Union[np.ndarray, pd.DataFrame, List[PredictionInput]]
ManyOutputsUnionType = Union[np.ndarray, pd.DataFrame, List[PredictionOutput]]

# trusty type names
trusty_type_map = {"i": "number", "O": "categorical", "f": "number", "b": "bool"}


# universal docstrings for functions that use these data conversions ===============================
def data_conversion_docstring(*keys):
    r"""Using a list of keys, add descriptions of accepted input and output datatypes to docstrings
    Each key contains two strings, one described the union type and the other describing the accepted
    objects as a plain-text, bulleted list. For each key passed, there must be two format arguments,
    for example:

    input_value : {}.
        The input value accepts: {}
    """
    keylist = []
    for k in keys:
        if k in _conversion_docstrings:
            keylist += _conversion_docstrings[k]
        else:
            raise ValueError(
                "{} not in valid conversion docstring keys: {}".format(
                    k, list(_conversion_docstrings.keys())
                )
            )

    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*keylist)
        return obj

    return dec


_conversion_docstrings = {
    "one_input": [
        "int, float, :class:`numpy.number`, List[Union[int, float, :class:`numpy.number`]], "
        ":class:`numpy.ndarray`, :class:`pandas.DataFrame`, :class:`pandas.Series`, "
        "List[:class:`Feature`], or :class:`PredictionInput`",
        """
        
            * If there's only a single input feature, an ``int``, ``float``, or any of the 
              `Numpy equivalents <https://numpy.org/doc/stable/user/basics.types.html>`_ 
              can be used.
            * A list of ``int``, ``float``, or any of the 
              `Numpy equivalents <https://numpy.org/doc/stable/user/basics.types.html>`_.
            * Numpy array of shape ``[1, n_features]`` or ``[n_features]``
            * Pandas DataFrame with 1 row and ``n_features`` columns
            * Pandas Series with `n_features` rows
            * A List of TrustyAI :class:`Feature`, as created by the :func:`~feature` function
            * A TrustyAI :class:`PredictionInput`
            
        """,
    ],
    "one_output": [
        "int, float, :class:`numpy.number`, List[Union[int, float, :class:`numpy.number`]], "
        ":class:`numpy.ndarray`, :class:`pandas.DataFrame`, :class:`pandas.Series`, "
        " List[:class:`Output`], or :class:`PredictionOutput`",
        """
        
            * If there's only a single output, an ``int``, ``float``, or any of the 
              `Numpy equivalents <https://numpy.org/doc/stable/user/basics.types.html>`_  
              can be used.
            * A list of ``int``, ``float``, or any of the 
              `Numpy equivalents <https://numpy.org/doc/stable/user/basics.types.html>`_.
            * Numpy array of shape ``[1, n_outputs]`` or ``[n_outputs]``
            * Pandas DataFrame with 1 row and ``n_outputs`` columns
            * Pandas Series with `n_outputs` rows
            * A List of TrustyAI :class:`Output`, as created by the :func:`~output` function
            * A TrustyAI :class:`PredictionOutput`
            
        """,
    ],
    "many_inputs": [
        ":class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`PredictionInput`]]",
        """
        
            * Numpy array of shape ``[n_rows, n_features]``
            * Pandas DataFrame with `n_rows` rows and `n_features` columns
            * A list of TrustyAI :class:`PredictionInput`
            
        """,
    ],
    "many_outputs": [
        ":class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`PredictionOutput`]]",
        """
        
            * Numpy array of shape ``[n_rows, n_outputs]``
            * Pandas DataFrame with `n_rows` rows and `n_outputs` columns
            * A list of TrustyAI :class:`PredictionOutput`
            
        """,
    ],
}


# === Domain Inserter ==============================================================================
def domain_insertion(
    undomained_input: PredictionInput, feature_domains: List[FeatureDomain]
):
    """Given a PredictionInput and a corresponding list of feature domains, where
    `len(feature_domains) == len(PredictionInput.getFeatures()`, return a PredictionInput
    where the ith feature has the ith domain. If the ith domain is `None`, no new domain
    information will be added to the feature, thus keeping previous domain information or
    keeping it fixed if none has been supplied"""
    assert len(undomained_input.getFeatures()) == len(
        feature_domains
    ), "input has {} features, but {} feature domains were passed".format(
        len(undomained_input.getFeatures()), len(feature_domains)
    )

    domained_features = []
    for i, f in enumerate(undomained_input.getFeatures()):
        if feature_domains[i] is None:
            domained_features.append(
                Feature(f.getName(), f.getType(), f.getValue(), True, f.getDomain())
            )
        else:
            if not isinstance(f.getDomain(), EmptyFeatureDomain):
                warning_msg = (
                    "The supplied feature domain at position {} is specifying a new "
                    "domain to previously domain'ed {}, this will overwrite the "
                    "previous domain with the new one.".format(i, f.toString())
                )
                warnings.warn(warning_msg)
            domained_features.append(
                Feature(
                    f.getName(), f.getType(), f.getValue(), False, feature_domains[i]
                )
            )
    return PredictionInput(domained_features)


# === input functions ==============================================================================
def one_input_convert(
    python_inputs: OneInputUnionType,
    feature_names: Optional[List[str]] = None,
    feature_domains: Optional[List[FeatureDomain]] = None,
) -> PredictionInput:
    """Convert an object of OneInputUnionType into a PredictionInput."""
    if isinstance(python_inputs, (int, float, np.number)):
        python_inputs = np.array([[python_inputs]])
        pi = numpy_to_prediction_object(
            python_inputs, trustyai.model.feature, names=feature_names
        )[0]
    elif isinstance(python_inputs, list) and all(
        (isinstance(x, (int, float, np.number)) for x in python_inputs)
    ):
        python_inputs = np.array(python_inputs).reshape(1, -1)
        pi = numpy_to_prediction_object(
            python_inputs, trustyai.model.feature, names=feature_names
        )[0]
    elif isinstance(python_inputs, np.ndarray):
        if len(python_inputs.shape) == 1:
            python_inputs = python_inputs.reshape(1, -1)
        pi = numpy_to_prediction_object(
            python_inputs, trustyai.model.feature, names=feature_names
        )[0]
    elif isinstance(python_inputs, pd.DataFrame):
        pi = df_to_prediction_object(python_inputs, trustyai.model.feature)[0]
    elif isinstance(python_inputs, pd.Series):
        pi = df_to_prediction_object(
            pd.DataFrame([python_inputs]), trustyai.model.feature
        )[0]
    elif isinstance(python_inputs, PredictionInput):
        pi = python_inputs
    else:
        # fallback case is List[Feature]
        pi = PredictionInput(python_inputs)

    if feature_domains is not None:
        pi = domain_insertion(pi, feature_domains)
    return pi


def many_inputs_convert(
    python_inputs: ManyInputsUnionType,
    feature_names: Optional[List[str]] = None,
    feature_domains: Optional[List[FeatureDomain]] = None,
) -> List[PredictionInput]:
    """Convert an object of ManyInputsUnionType into a List[PredictionInput]"""
    if isinstance(python_inputs, np.ndarray):
        if len(python_inputs.shape) == 1:
            python_inputs = python_inputs.reshape(1, -1)
        pis = numpy_to_prediction_object(
            python_inputs, trustyai.model.feature, names=feature_names
        )
    elif isinstance(python_inputs, pd.DataFrame):
        pis = df_to_prediction_object(python_inputs, trustyai.model.feature)
    else:
        # fallback case is List[PredictionInput]
        pis = python_inputs

    if feature_domains is not None:
        pis = [domain_insertion(pi, feature_domains) for pi in pis]
    return pis


# === output functions =============================================================================
def one_output_convert(
    python_outputs: OneOutputUnionType, names: Optional[List[str]] = None
) -> PredictionOutput:
    """Convert an object of OneOutputUnionType into a PredictionOutput"""
    if isinstance(python_outputs, (int, np.integer, float, np.inexact)):
        python_outputs = np.array([[python_outputs]])
        po = numpy_to_prediction_object(
            python_outputs, trustyai.model.output, names=names
        )[0]
    elif isinstance(python_outputs, list) and all(
        (isinstance(x, (int, float, np.number)) for x in python_outputs)
    ):
        python_outputs = np.array(python_outputs).reshape(1, -1)
        po = numpy_to_prediction_object(
            python_outputs, trustyai.model.output, names=names
        )[0]
    elif isinstance(python_outputs, np.ndarray):
        if len(python_outputs.shape) == 1:
            python_outputs = python_outputs.reshape(1, -1)
        po = numpy_to_prediction_object(
            python_outputs, trustyai.model.output, names=names
        )[0]
    elif isinstance(python_outputs, pd.DataFrame):
        po = df_to_prediction_object(python_outputs, trustyai.model.output)[0]
    elif isinstance(python_outputs, pd.Series):
        po = df_to_prediction_object(
            pd.DataFrame([python_outputs]), trustyai.model.output
        )[0]
    elif isinstance(python_outputs, PredictionOutput):
        po = python_outputs
    else:
        # fallback is List[Output]
        po = PredictionOutput(python_outputs)
    return po


def many_outputs_convert(
    python_outputs: ManyOutputsUnionType, names: Optional[List[str]] = None
) -> List[PredictionOutput]:
    """Convert an object of ManyOutputsUnionType into a List[PredictionOutput]"""
    if isinstance(python_outputs, np.ndarray):
        if len(python_outputs.shape) == 1:
            python_outputs = python_outputs.reshape(1, -1)
        return numpy_to_prediction_object(
            python_outputs, trustyai.model.output, names=names
        )
    if isinstance(python_outputs, pd.DataFrame):
        return df_to_prediction_object(python_outputs, trustyai.model.output)
    # fallback case is List[PredictionOutput]
    return python_outputs


# === TrustyAI Conversions =========================================================================
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
    wrapper = PredictionInput if func is trustyai.model.feature else PredictionOutput

    for row in typed_rows:
        values = list(row)
        collection = []
        for fv in values:
            f = func(name=fv[2], dtype=fv[1], value=fv[0])
            collection.append(f)
        predictions.append(wrapper(collection))
    return predictions


def numpy_to_prediction_object(
    array: np.ndarray, func, names=None
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
    names: list{str]
        The names of the features/outputs in the created PredictionInput/PredictionOutput object
    """
    shape = array.shape

    # pylint: disable=comparison-with-callable
    if func == trustyai.model.feature:
        prefix = "input"
        wrapper = PredictionInput
    else:
        prefix = "output"
        wrapper = PredictionOutput
    if names is None:
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


def prediction_object_to_numpy(
    objects: Union[List[PredictionInput], List[PredictionOutput]],
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


def prediction_object_to_pandas(
    objects: Union[List[PredictionInput], List[PredictionOutput]],
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


def __partition_column_indices(
    size: int, outputs: Optional[List[int]] = None
) -> Tuple[List[int], List[int]]:
    indices = list(range(size))
    if not outputs:  # If no output column supplied, assume the right-most
        output_indices = [size - 1]
        input_indices = list(filterfalse(output_indices.__contains__, indices))
    else:
        output_indices = outputs
        input_indices = list(filterfalse(outputs.__contains__, indices))
    return input_indices, output_indices


def to_trusty_dataframe(
    data: Union[pd.DataFrame, np.ndarray],
    outputs: Optional[List[int]] = None,
    no_outputs=False,
    feature_names: Optional[List[str]] = None,
) -> Dataframe:
    """Convert Pandas dataframes or NumPy arrays into TrustyAI dataframes"""
    if isinstance(data, pd.DataFrame):
        return df_to_trusty_dataframe(
            data=data,
            outputs=outputs,
            no_outputs=no_outputs,
            feature_names=feature_names,
        )
    if isinstance(data, np.ndarray):
        return numpy_to_trusty_dataframe(
            arr=data,
            outputs=outputs,
            no_outputs=no_outputs,
            feature_names=feature_names,
        )

    raise ValueError("Only Pandas dataframes and NumPy arrays supported at the moment.")


def df_to_trusty_dataframe(
    data: pd.DataFrame,
    outputs: Optional[List[int]] = None,
    no_outputs=False,
    feature_names: Optional[List[str]] = None,
) -> Dataframe:
    """
    Converts a Pandas :class:`pandas.DataFrame` into a TrustyAI :class:`Dataframe`.
    Either outputs can be provided as a list of column indices or `no_outputs` can be specified, for an inputs-only
    :class:`Dataframe`.

    Parameters
    ----------
    outputs : List[int]
        Optional list of column indices to be marked as outputs

    no_outputs : bool
        Specify if the :class:`Dataframe` is inputs-only

    feature_names : Optional[List[str]]
        Optional list of feature names. If not provided, the Pandas dataframe column names will be used
    """
    data = data.reset_index(drop=True)
    n_columns = len(data.columns)
    if not no_outputs:

        input_indices, output_indices = __partition_column_indices(n_columns, outputs)

        if feature_names:
            input_names = [feature_names[i] for i in input_indices]
            output_names = [feature_names[i] for i in output_indices]
        else:
            input_names = None
            output_names = None

        pi = many_inputs_convert(
            python_inputs=data.iloc[:, input_indices], feature_names=input_names
        )
        po = many_outputs_convert(
            python_outputs=data.iloc[:, output_indices], names=output_names
        )

        return Dataframe.createFrom(pi, po)

    pi = many_inputs_convert(data)
    return Dataframe.createFromInputs(pi)


def numpy_to_trusty_dataframe(
    arr: np.ndarray,
    feature_names: List[str],
    outputs: Optional[List[int]] = None,
    no_outputs=False,
) -> Dataframe:
    """
    Converts a NumPy :class:`np.ndarray` into a TrustyAI :class:`Dataframe`.
    Either outputs can be provided as a list of column indices or `no_outputs` can be specified, for an inputs-only
    :class:`Dataframe`.

    Parameters
    ----------
    outputs : List[int]
        Optional list of column indices to be marked as outputs

    no_outputs : bool
        Specify if the :class:`Dataframe` is inputs-only

    feature_names : Optional[List[str]]
        Optional list of feature names. If not provided, the Pandas dataframe column names will be used
    """
    n_columns = arr.shape[1]
    if not no_outputs:
        input_indices, output_indices = __partition_column_indices(n_columns, outputs)

        input_names = [feature_names[i] for i in input_indices]
        output_names = [feature_names[i] for i in output_indices]
        axis = 1

        pi = many_inputs_convert(
            python_inputs=np.take(arr, input_indices, axis), feature_names=input_names
        )
        po = many_outputs_convert(
            python_outputs=np.take(arr, output_indices, axis), names=output_names
        )

        return Dataframe.createFrom(pi, po)

    pi = many_inputs_convert(arr)
    return Dataframe.createFromInputs(pi)
