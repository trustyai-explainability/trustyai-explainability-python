"""Group fairness metrics"""

# pylint: disable = import-error
from typing import List, Optional, Any, Union

import numpy as np
import pandas as pd
from jpype import JInt
from org.kie.trustyai.metrics.fairness.group import (
    DisparateImpactRatio,
    GroupStatisticalParityDifference,
    GroupAverageOddsDifference,
    GroupAveragePredictiveValueDifference,
)

from trustyai.model import Value, PredictionProvider, Model
from trustyai.utils.data_conversions import (
    OneOutputUnionType,
    one_output_convert,
    to_trusty_dataframe,
)

ColumSelector = Union[List[int], List[str]]


def _column_selector_to_index(columns: ColumSelector, dataframe: pd.DataFrame):
    """Returns a list of input and output indices, given an index size and output indices"""
    if len(columns) == 0:
        raise ValueError("Must specify at least one column")

    if isinstance(columns[0], str):  # passing column
        columns = dataframe.columns.get_indexer(columns)
    indices = [JInt(c) for c in columns]  # Java casting
    return indices


def statistical_parity_difference(
    privileged: Union[pd.DataFrame, np.ndarray],
    unprivileged: Union[pd.DataFrame, np.ndarray],
    favorable: OneOutputUnionType,
    outputs: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
) -> float:
    """Calculate Statistical Parity Difference between privileged and unprivileged dataframes"""
    favorable_prediction_object = one_output_convert(favorable)
    return GroupStatisticalParityDifference.calculate(
        to_trusty_dataframe(
            data=privileged, outputs=outputs, feature_names=feature_names
        ),
        to_trusty_dataframe(
            data=unprivileged, outputs=outputs, feature_names=feature_names
        ),
        favorable_prediction_object.outputs,
    )


# pylint: disable = line-too-long, too-many-arguments
def statistical_parity_difference_model(
    samples: Union[pd.DataFrame, np.ndarray],
    model: Union[PredictionProvider, Model],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    favorable: OneOutputUnionType,
    feature_names: Optional[List[str]] = None,
) -> float:
    """Calculate Statistical Parity Difference using a samples dataframe and a model"""
    favorable_prediction_object = one_output_convert(favorable)
    _privilege_values = [Value(v) for v in privilege_values]
    _jsamples = to_trusty_dataframe(
        data=samples, no_outputs=True, feature_names=feature_names
    )
    return GroupStatisticalParityDifference.calculate(
        _jsamples,
        model,
        _column_selector_to_index(privilege_columns, samples),
        _privilege_values,
        favorable_prediction_object.outputs,
    )


def disparate_impact_ratio(
    privileged: Union[pd.DataFrame, np.ndarray],
    unprivileged: Union[pd.DataFrame, np.ndarray],
    favorable: OneOutputUnionType,
    outputs: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
) -> float:
    """Calculate Disparate Impact Ration between privileged and unprivileged dataframes"""
    favorable_prediction_object = one_output_convert(favorable)
    return DisparateImpactRatio.calculate(
        to_trusty_dataframe(
            data=privileged, outputs=outputs, feature_names=feature_names
        ),
        to_trusty_dataframe(
            data=unprivileged, outputs=outputs, feature_names=feature_names
        ),
        favorable_prediction_object.outputs,
    )


# pylint: disable = line-too-long
def disparate_impact_ratio_model(
    samples: Union[pd.DataFrame, np.ndarray],
    model: Union[PredictionProvider, Model],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    favorable: OneOutputUnionType,
    feature_names: Optional[List[str]] = None,
) -> float:
    """Calculate Disparate Impact Ration using a samples dataframe and a model"""
    favorable_prediction_object = one_output_convert(favorable)
    _privilege_values = [Value(v) for v in privilege_values]
    _jsamples = to_trusty_dataframe(
        data=samples, no_outputs=True, feature_names=feature_names
    )
    return DisparateImpactRatio.calculate(
        _jsamples,
        model,
        _column_selector_to_index(privilege_columns, samples),
        _privilege_values,
        favorable_prediction_object.outputs,
    )


# pylint: disable = too-many-arguments
def average_odds_difference(
    test: Union[pd.DataFrame, np.ndarray],
    truth: Union[pd.DataFrame, np.ndarray],
    privilege_columns: ColumSelector,
    privilege_values: OneOutputUnionType,
    positive_class: List[Any],
    outputs: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
) -> float:
    """Calculate Average Odds between two dataframes"""
    if test.shape != truth.shape:
        raise ValueError(
            f"Dataframes have different shapes ({test.shape} and {truth.shape})"
        )
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    # determine privileged columns
    _privilege_columns = _column_selector_to_index(privilege_columns, test)
    return GroupAverageOddsDifference.calculate(
        to_trusty_dataframe(data=test, outputs=outputs, feature_names=feature_names),
        to_trusty_dataframe(data=truth, outputs=outputs, feature_names=feature_names),
        _privilege_columns,
        _privilege_values,
        _positive_class,
    )


def average_odds_difference_model(
    samples: Union[pd.DataFrame, np.ndarray],
    model: Union[PredictionProvider, Model],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    positive_class: List[Any],
    feature_names: Optional[List[str]] = None,
) -> float:
    """Calculate Average Odds for a sample dataframe using the provided model"""
    _jsamples = to_trusty_dataframe(
        data=samples, no_outputs=True, feature_names=feature_names
    )
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    # determine privileged columns
    _privilege_columns = _column_selector_to_index(privilege_columns, samples)
    return GroupAverageOddsDifference.calculate(
        _jsamples, model, _privilege_columns, _privilege_values, _positive_class
    )


def average_predictive_value_difference(
    test: Union[pd.DataFrame, np.ndarray],
    truth: Union[pd.DataFrame, np.ndarray],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    positive_class: List[Any],
    outputs: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
) -> float:
    """Calculate Average Predictive Value Difference between two dataframes"""
    if test.shape != truth.shape:
        raise ValueError(
            f"Dataframes have different shapes ({test.shape} and {truth.shape})"
        )
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    _privilege_columns = _column_selector_to_index(privilege_columns, test)
    return GroupAveragePredictiveValueDifference.calculate(
        to_trusty_dataframe(data=test, outputs=outputs, feature_names=feature_names),
        to_trusty_dataframe(data=truth, outputs=outputs, feature_names=feature_names),
        _privilege_columns,
        _privilege_values,
        _positive_class,
    )


# pylint: disable = line-too-long
def average_predictive_value_difference_model(
    samples: Union[pd.DataFrame, np.ndarray],
    model: Union[PredictionProvider, Model],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    positive_class: List[Any],
) -> float:
    """Calculate Average Predictive Value Difference for a sample dataframe using the provided model"""
    _jsamples = to_trusty_dataframe(samples, no_outputs=True)
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    # determine privileged columns
    _privilege_columns = _column_selector_to_index(privilege_columns, samples)
    return GroupAveragePredictiveValueDifference.calculate(
        _jsamples, model, _privilege_columns, _privilege_values, _positive_class
    )
