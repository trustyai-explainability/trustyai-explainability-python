"""Group fairness metrics"""
# pylint: disable = import-error
from typing import List, Optional, Any, Union

import pandas as pd
from jpype import JInt
from org.kie.trustyai.explainability.metrics import FairnessMetrics

from trustyai.model import Output, Value, PredictionProvider, Model
from trustyai.utils.data_conversions import pandas_to_trusty

ColumSelector = Union[List[int], List[str]]


def _column_selector_to_index(columns: ColumSelector, dataframe: pd.DataFrame):
    if isinstance(columns[0], str):  # passing column
        columns = dataframe.columns.get_indexer(columns)
    indices = [JInt(c) for c in columns]  # Java casting
    return indices


def statistical_parity_difference(
    privileged: pd.DataFrame,
    unprivileged: pd.DataFrame,
    favorable: List[Output],
    outputs: Optional[List[int]] = None,
) -> float:
    """Calculate Statistical Parity Difference between privileged and unprivileged dataframes"""
    return FairnessMetrics.groupStatisticalParityDifference(
        pandas_to_trusty(privileged, outputs),
        pandas_to_trusty(unprivileged, outputs),
        favorable,
    )


# pylint: disable = line-too-long
def statistical_parity_difference_model(
    samples: pd.DataFrame,
    model: Union[PredictionProvider, Model],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    favorable: List[Output],
) -> float:
    """Calculate Statistical Parity Difference using a samples dataframe and a model"""
    _privilege_values = [Value(v) for v in privilege_values]
    _jsamples = pandas_to_trusty(samples, no_outputs=True)
    return FairnessMetrics.groupStatisticalParityDifference(
        _jsamples,
        model,
        _column_selector_to_index(privilege_columns, samples),
        _privilege_values,
        favorable,
    )


def disparate_impact_ratio(
    privileged: pd.DataFrame,
    unprivileged: pd.DataFrame,
    favorable: List[Output],
    outputs: Optional[List[int]] = None,
) -> float:
    """Calculate Disparate Impact Ration between privileged and unprivileged dataframes"""
    return FairnessMetrics.groupDisparateImpactRatio(
        pandas_to_trusty(privileged, outputs),
        pandas_to_trusty(unprivileged, outputs),
        favorable,
    )


# pylint: disable = line-too-long
def disparate_impact_ratio_model(
    samples: pd.DataFrame,
    model: Union[PredictionProvider, Model],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    favorable: List[Output],
) -> float:
    """Calculate Disparate Impact Ration using a samples dataframe and a model"""
    _privilege_values = [Value(v) for v in privilege_values]
    _jsamples = pandas_to_trusty(samples, no_outputs=True)
    return FairnessMetrics.groupDisparateImpactRatio(
        _jsamples,
        model,
        _column_selector_to_index(privilege_columns, samples),
        _privilege_values,
        favorable,
    )


# pylint: disable = too-many-arguments
def average_odds_difference(
    test: pd.DataFrame,
    truth: pd.DataFrame,
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    positive_class: List[Any],
    outputs: Optional[List[int]] = None,
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
    return FairnessMetrics.groupAverageOddsDifference(
        pandas_to_trusty(test, outputs),
        pandas_to_trusty(truth, outputs),
        _privilege_columns,
        _privilege_values,
        _positive_class,
    )


def average_odds_difference_model(
    samples: pd.DataFrame,
    model: Union[PredictionProvider, Model],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    positive_class: List[Any],
) -> float:
    """Calculate Average Odds for a sample dataframe using the provided model"""
    _jsamples = pandas_to_trusty(samples, no_outputs=True)
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    # determine privileged columns
    _privilege_columns = _column_selector_to_index(privilege_columns, samples)
    return FairnessMetrics.groupAverageOddsDifference(
        _jsamples, model, _privilege_columns, _privilege_values, _positive_class
    )


def average_predictive_value_difference(
    test: pd.DataFrame,
    truth: pd.DataFrame,
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    positive_class: List[Any],
    outputs: Optional[List[int]] = None,
) -> float:
    """Calculate Average Predictive Value Difference between two dataframes"""
    if test.shape != truth.shape:
        raise ValueError(
            f"Dataframes have different shapes ({test.shape} and {truth.shape})"
        )
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    _privilege_columns = _column_selector_to_index(privilege_columns, test)
    return FairnessMetrics.groupAveragePredictiveValueDifference(
        pandas_to_trusty(test, outputs),
        pandas_to_trusty(truth, outputs),
        _privilege_columns,
        _privilege_values,
        _positive_class,
    )


# pylint: disable = line-too-long
def average_predictive_value_difference_model(
    samples: pd.DataFrame,
    model: Union[PredictionProvider, Model],
    privilege_columns: ColumSelector,
    privilege_values: List[Any],
    positive_class: List[Any],
) -> float:
    """Calculate Average Predictive Value Difference for a sample dataframe using the provided model"""
    _jsamples = pandas_to_trusty(samples, no_outputs=True)
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    # determine privileged columns
    _privilege_columns = _column_selector_to_index(privilege_columns, samples)
    return FairnessMetrics.groupAveragePredictiveValueDifference(
        _jsamples, model, _privilege_columns, _privilege_values, _positive_class
    )
