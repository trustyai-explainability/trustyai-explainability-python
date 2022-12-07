# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Fairness metrics test suite"""
from typing import List, Optional

from common import *

from pytest import approx
import pandas as pd
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
import os
import pathlib

from trustyai.metrics.fairness.group import statistical_parity_difference, disparate_impact_ratio, \
    average_odds_difference, average_predictive_value_difference, statistical_parity_difference_model, \
    average_odds_difference_model, average_predictive_value_difference_model
from trustyai.model import output, Model
from java.util import Random

jrandom = Random()

TEST_DIR = pathlib.Path(__file__).parent.resolve()

INCOME_DF_BIASED = pd.read_csv(os.path.join(TEST_DIR, "data/income-biased.zip"), index_col=False)
INCOME_DF_UNBIASED = pd.read_csv(
    os.path.join(TEST_DIR, "data/income-unbiased.zip"), index_col=False)
XGB_MODEL = XGBClassifier()
XGB_MODEL.load_model(os.path.join(TEST_DIR, "models/income-xgb-biased.ubj"))


def create_random_dataframe(weights: Optional[List[float]] = None):
    if not weights:
        weights = [0.9, 0.1]
    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=2, class_sep=2, flip_y=0, weights=weights,
                               random_state=23)

    return pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'y': y
    })


def test_statistical_parity_difference_random():
    """Test Statistical Parity Difference (unbalanced random data)"""

    df = create_random_dataframe()

    privileged = df[df.x1 < 0]
    unprivileged = df[df.x1 >= 0]
    favorable = output("y", dtype="number", value=1)
    score = statistical_parity_difference(privileged, unprivileged, [favorable])
    assert score == approx(0.9, 0.09)


def test_statistical_parity_difference_income():
    """Test Statistical Parity Difference (income data)"""

    df = INCOME_DF_BIASED.copy()

    privileged = df[df.gender == 1]
    unprivileged = df[df.gender == 0]
    favorable = output("income", dtype="number", value=1)
    score = statistical_parity_difference(privileged, unprivileged, [favorable])
    assert score == approx(-0.15, abs=0.01)


def test_statistical_parity_difference_model():
    """Test Statistical Parity Difference (XGBoost model)"""

    df = INCOME_DF_BIASED.copy()
    X = df[["age", "race", "gender"]]

    favorable = output("income", dtype="number", value=1)
    model = Model(XGB_MODEL.predict, dataframe_input=True, output_names=["approved"])
    score = statistical_parity_difference_model(X, model, [2], [1], [favorable])
    assert score == approx(0.0, abs=0.09)


def test_disparate_impact_ratio_random():
    """Test Disparate Impact Ratio (unbalanced random data)"""

    df = create_random_dataframe(weights=[0.5, 0.5])

    privileged = df[df.x1 < 0]
    unprivileged = df[df.x1 >= 0]
    favorable = output("y", dtype="number", value=1)
    score = disparate_impact_ratio(privileged, unprivileged, [favorable])
    assert score == approx(130.0, abs=5.0)


def test_disparate_impact_ratio_income():
    """Test Disparate Impact Ratio (income data)"""

    df = INCOME_DF_BIASED.copy()

    privileged = df[df.gender == 1]
    unprivileged = df[df.gender == 0]
    favorable = output("income", dtype="number", value=1)
    score = disparate_impact_ratio(privileged, unprivileged, [favorable])
    assert score == approx(0.4, abs=0.05)


def test_average_odds_difference():
    """Test Average Odds Difference (unbalanced random data)"""
    PRIVILEGED_CLASS_GENDER = 1
    UNPRIVILEGED_CLASS_GENDER = 0
    PRIVILEGED_CLASS_RACE = 4
    UNPRIVILEGED_CLASS_RACE = 2
    score = average_odds_difference(INCOME_DF_BIASED, INCOME_DF_UNBIASED, [1, 2],
                                    [PRIVILEGED_CLASS_RACE, PRIVILEGED_CLASS_GENDER], [1], [3])
    assert score == approx(0.12, abs=0.1)
    score = average_odds_difference(INCOME_DF_BIASED, INCOME_DF_UNBIASED, [1, 2],
                                    [UNPRIVILEGED_CLASS_RACE, UNPRIVILEGED_CLASS_GENDER], [1], [3])
    assert score == approx(0.2, abs=0.1)


def test_average_odds_difference_model():
    """Test Average Odds Difference (XGBoost income model)"""
    df = INCOME_DF_BIASED.copy()
    X = df[["age", "race", "gender"]]

    favorable = output("income", dtype="number", value=1)
    model = Model(XGB_MODEL.predict, dataframe_input=True, output_names=["approved"])

    score = average_odds_difference_model(samples=X,
                                          model=model,
                                          privilege_columns=[2],
                                          privilege_values=[1],
                                          positive_class=[1])

    assert score == approx(0.0, abs=0.09)
    score = average_odds_difference_model(samples=X,
                                          model=model,
                                          privilege_columns=[2],
                                          privilege_values=[0],
                                          positive_class=[1])

    assert score == approx(0.0, abs=0.09)


def test_average_predictive_value_difference():
    """Test Average Predictive Value Difference (unbalanced random data)"""
    PRIVILEGED_CLASS_GENDER = 1
    UNPRIVILEGED_CLASS_GENDER = 0
    PRIVILEGED_CLASS_RACE = 4
    UNPRIVILEGED_CLASS_RACE = 2
    score = average_predictive_value_difference(INCOME_DF_BIASED, INCOME_DF_UNBIASED, [1, 2],
                                                [PRIVILEGED_CLASS_RACE, PRIVILEGED_CLASS_GENDER], [1], [3])
    assert score == approx(-0.3, abs=0.1)
    score = average_predictive_value_difference(INCOME_DF_BIASED, INCOME_DF_UNBIASED, [1, 2],
                                                [UNPRIVILEGED_CLASS_RACE, UNPRIVILEGED_CLASS_GENDER], [1], [3])
    assert score == approx(-0.22, abs=0.05)


def test_average_predictive_value_difference_model():
    """Test Average Predictive Value Difference (XGB income model)"""

    df = INCOME_DF_BIASED.copy()
    X = df[["age", "race", "gender"]]

    model = Model(XGB_MODEL.predict, dataframe_input=True, output_names=["approved"])

    score = average_predictive_value_difference_model(samples=X,
                                                      model=model,
                                                      privilege_columns=[2],
                                                      privilege_values=[1],
                                                      positive_class=[1])

    assert score == approx(0.0, abs=0.09)
    score = average_predictive_value_difference_model(samples=X,
                                                      model=model,
                                                      privilege_columns=[2],
                                                      privilege_values=[0],
                                                      positive_class=[1])

    assert score == approx(0.0, abs=0.09)
