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


def statistical_parity_difference(priviledged: pd.DataFrame,
                                  unpriviledged: pd.DataFrame,
                                  favorable: List[Output],
                                  outputs: Optional[List[int]] = None) -> float:
    return FairnessMetrics.groupStatisticalParityDifference(pandas_to_trusty(priviledged, outputs),
                                                            pandas_to_trusty(unpriviledged, outputs),
                                                            favorable)


def statistical_parity_difference_model(samples: pd.DataFrame,
                                        model: Union[PredictionProvider, Model],
                                        privilege_columns: ColumSelector,
                                        privilege_values: List[Any],
                                        favorable: List[Output]) -> float:
    _privilege_values = [Value(v) for v in privilege_values]
    _jsamples = pandas_to_trusty(samples, no_outputs=True)
    return FairnessMetrics.groupStatisticalParityDifference(_jsamples,
                                                            model,
                                                            _column_selector_to_index(privilege_columns, samples),
                                                            _privilege_values,
                                                            favorable)


def disparate_impact_ratio(priviledged: pd.DataFrame,
                           unpriviledged: pd.DataFrame,
                           favorable: List[Output],
                           outputs: Optional[List[int]] = None) -> float:
    return FairnessMetrics.groupDisparateImpactRatio(pandas_to_trusty(priviledged, outputs),
                                                     pandas_to_trusty(unpriviledged, outputs),
                                                     favorable)


def disparate_impact_ratio_model(samples: pd.DataFrame,
                                 model: Union[PredictionProvider, Model],
                                 privilege_columns: ColumSelector,
                                 privilege_values: List[Any],
                                 favorable: List[Output]) -> float:
    _privilege_values = [Value(v) for v in privilege_values]
    _jsamples = pandas_to_trusty(samples, no_outputs=True)
    return FairnessMetrics.groupDisparateImpactRatio(_jsamples,
                                                     model,
                                                     _column_selector_to_index(privilege_columns, samples),
                                                     _privilege_values,
                                                     favorable)


def average_odds_difference(test: pd.DataFrame,
                            truth: pd.DataFrame,
                            privilege_columns: ColumSelector,
                            privilege_values: List[Any],
                            positive_class: List[Any],
                            outputs: Optional[List[int]] = None) -> float:
    if test.shape != truth.shape:
        raise ValueError(f"Dataframes have different shapes ({test.shape} and {truth.shape})")
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    # determine privileged columns
    _privilege_columns = _column_selector_to_index(privilege_columns, test)
    return FairnessMetrics.groupAverageOddsDifference(pandas_to_trusty(test, outputs),
                                                      pandas_to_trusty(truth, outputs),
                                                      _privilege_columns,
                                                      _privilege_values,
                                                      _positive_class)


def average_odds_difference_model(samples: pd.DataFrame,
                                  model: Union[PredictionProvider, Model],
                                  privilege_columns: ColumSelector,
                                  privilege_values: List[Any],
                                  positive_class: List[Any]) -> float:
    _jsamples = pandas_to_trusty(samples, no_outputs=True)
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    # determine privileged columns
    _privilege_columns = _column_selector_to_index(privilege_columns, samples)
    return FairnessMetrics.groupAverageOddsDifference(_jsamples,
                                                      model,
                                                      _privilege_columns,
                                                      _privilege_values,
                                                      _positive_class)


def average_predictive_value_difference(test: pd.DataFrame,
                                        truth: pd.DataFrame,
                                        privilege_columns: ColumSelector,
                                        privilege_values: List[Any],
                                        positive_class: List[Any],
                                        outputs: Optional[List[int]] = None) -> float:
    if test.shape != truth.shape:
        raise ValueError(f"Dataframes have different shapes ({test.shape} and {truth.shape})")
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    _privilege_columns = _column_selector_to_index(privilege_columns, test)
    return FairnessMetrics.groupAveragePredictiveValueDifference(pandas_to_trusty(test, outputs),
                                                                 pandas_to_trusty(truth, outputs),
                                                                 _privilege_columns,
                                                                 _privilege_values,
                                                                 _positive_class)


def average_predictive_value_difference_model(samples: pd.DataFrame,
                                              model: Union[PredictionProvider, Model],
                                              privilege_columns: ColumSelector,
                                              privilege_values: List[Any],
                                              positive_class: List[Any]) -> float:
    _jsamples = pandas_to_trusty(samples, no_outputs=True)
    _privilege_values = [Value(v) for v in privilege_values]
    _positive_class = [Value(v) for v in positive_class]
    # determine privileged columns
    _privilege_columns = _column_selector_to_index(privilege_columns, samples)
    return FairnessMetrics.groupAveragePredictiveValueDifference(_jsamples,
                                                                 model,
                                                                 _privilege_columns,
                                                                 _privilege_values,
                                                                 _positive_class)
