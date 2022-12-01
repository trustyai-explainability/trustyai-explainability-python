"""Data Converters between Python and Java"""
# pylint: disable = import-error, line-too-long, trailing-whitespace, unused-import, cyclic-import
# pylint: disable = consider-using-f-string, invalid-name, wrong-import-order

from typing import Union, List

import trustyai.model
from trustyai.model.domain import feature_domain
from org.kie.trustyai.explainability.model import (
    Feature,
    Output,
    PredictionInput,
    PredictionOutput,
)
from org.kie.trustyai.explainability.model.domain import FeatureDomain

import pandas as pd
import numpy as np

# UNION TYPES FOR INPUTS AND OUTPUTS
# if a TrustyAI function wants AN input/output, it should accept this union type:
OneInputUnionType = Union[
    np.ndarray, pd.DataFrame, pd.Series, List[Feature], PredictionInput
]
OneOutputUnionType = Union[
    np.ndarray, pd.DataFrame, pd.Series, List[Output], PredictionOutput
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
        ":class:`numpy.ndarray`, :class:`pandas.DataFrame`, :class:`pandas.Series`, List[:class:`Feature`], or :class:`PredictionInput`",
        """
        
            * Numpy array of shape ``[1, n_features]`` or ``[n_features]``
            * Pandas DataFrame with 1 row and ``n_features`` columns
            * Pandas Series with `n_features` rows
            * A List of TrustyAI :class:`Feature`, as created by the :func:`~feature` function
            * A TrustyAI :class:`PredictionInput`
            
        """,
    ],
    "one_output": [
        ":class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Output`], or :class:`PredictionOutput`",
        """
        
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
    where the ith feature has the ith domain. If the ith domain is `None`, the feature
    is constrained."""
    assert len(undomained_input.getFeatures()) == len(feature_domains)

    domained_features = []
    for i, f in enumerate(undomained_input.getFeatures()):
        if feature_domains[i] is None:
            domained_features.append(
                Feature(f.getName(), f.getType(), f.getValue(), True, None)
            )
        else:
            domained_features.append(
                Feature(
                    f.getName(), f.getType(), f.getValue(), False, feature_domains[i]
                )
            )
    return PredictionInput(domained_features)


# === input functions ==============================================================================
def one_input_convert(
    python_inputs: OneInputUnionType, feature_domains: FeatureDomain = None
) -> PredictionInput:
    """Convert an object of OneInputUnionType into a PredictionInput."""
    if isinstance(python_inputs, np.ndarray):
        if len(python_inputs.shape) == 1:
            python_inputs = python_inputs.reshape(1, -1)
        pi = numpy_to_prediction_object(python_inputs, trustyai.model.feature)[0]
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
    python_inputs: ManyInputsUnionType, feature_domains: List[FeatureDomain] = None
) -> List[PredictionInput]:
    """Convert an object of ManyInputsUnionType into a List[PredictionInput]"""
    if isinstance(python_inputs, np.ndarray):
        if len(python_inputs.shape) == 1:
            python_inputs = python_inputs.reshape(1, -1)
        pis = numpy_to_prediction_object(python_inputs, trustyai.model.feature)
    elif isinstance(python_inputs, pd.DataFrame):
        pis = df_to_prediction_object(python_inputs, trustyai.model.feature)
    else:
        # fallback case is List[PredictionInput]
        pis = python_inputs

    if feature_domains is not None:
        pis = [domain_insertion(pi, feature_domains) for pi in pis]
    return pis


# === output functions =============================================================================
def one_output_convert(python_outputs: OneOutputUnionType) -> PredictionOutput:
    """Convert an object of OneOutputUnionType into a PredictionOutput"""
    if isinstance(python_outputs, np.ndarray):
        if len(python_outputs.shape) == 1:
            python_outputs = python_outputs.reshape(1, -1)
        return numpy_to_prediction_object(python_outputs, trustyai.model.output)[0]
    if isinstance(python_outputs, pd.DataFrame):
        return df_to_prediction_object(python_outputs, trustyai.model.output)[0]
    if isinstance(python_outputs, pd.Series):
        return df_to_prediction_object(
            pd.DataFrame([python_outputs]), trustyai.model.output
        )[0]
    if isinstance(python_outputs, PredictionOutput):
        return python_outputs
    # fallback is List[Output]
    return PredictionOutput(python_outputs)


def many_outputs_convert(
    python_outputs: ManyOutputsUnionType,
) -> List[PredictionOutput]:
    """Convert an object of ManyOutputsUnionType into a List[PredictionOutput]"""
    if isinstance(python_outputs, np.ndarray):
        if len(python_outputs.shape) == 1:
            python_outputs = python_outputs.reshape(1, -1)
        return numpy_to_prediction_object(python_outputs, trustyai.model.output)
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
