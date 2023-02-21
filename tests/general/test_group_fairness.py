# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Fairness metrics test suite"""
from typing import List, Optional

from common import *

from pytest import approx
import pandas as pd
from xgboost import XGBClassifier
import os
import joblib
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
# XGB_MODEL = XGBClassifier()
XGB_MODEL = joblib.load(os.path.join(TEST_DIR, "models/income-xgd-biased.joblib"))


def test_statistical_parity_difference_random():
    """Test Statistical Parity Difference (unbalanced random data)"""

    df = create_random_dataframe()

    privileged = df[df.x1 < 0]
    unprivileged = df[df.x1 >= 0]
    favorable = output("y", dtype="number", value=1)
    score = statistical_parity_difference(privileged, unprivileged, [favorable])
    assert score == approx(0.9, 0.09)


def test_statistical_parity_difference_random_numpy():
    """Test Statistical Parity Difference (unbalanced random data, NumPy)"""

    data = create_random_dataframe().to_numpy()

    privileged = data[np.where(data[:, 0] < 0)]
    unprivileged = data[np.where(data[:, 0] >= 0)]
    favorable = output("y", dtype="number", value=1)
    score = statistical_parity_difference(privileged=privileged,
                                          unprivileged=unprivileged,
                                          favorable=[favorable],
                                          feature_names=['x1', 'x2', 'y'])
    assert score == approx(0.0, 0.09)


def test_statistical_parity_difference_income():
    """Test Statistical Parity Difference (income data)"""

    df = INCOME_DF_BIASED.copy()

    privileged = df[df.gender == 1]
    unprivileged = df[df.gender == 0]
    favorable = output("income", dtype="number", value=1)
    score = statistical_parity_difference(privileged, unprivileged, [favorable])
    assert score == approx(-0.15, abs=0.01)


def test_statistical_parity_difference_income_numpy():
    """Test Statistical Parity Difference (income data, NumPy)"""

    arr = INCOME_DF_BIASED.to_numpy()

    privileged = arr[np.where(arr[:, 2] == 1)]
    unprivileged = arr[np.where(arr[:, 2] == 0)]
    favorable = output("income", dtype="number", value=1)
    score = statistical_parity_difference(privileged=privileged,
                                          unprivileged=unprivileged,
                                          favorable=[favorable],
                                          feature_names=['age', 'race', 'gender', 'income'])
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


def test_disparate_impact_ratio_income_numpy():
    """Test Disparate Impact Ratio (income data, NumPy)"""

    data = INCOME_DF_BIASED.to_numpy()

    privileged = data[np.where(data[:, 2] == 1)]
    unprivileged = data[np.where(data[:, 2] == 0)]
    favorable = output("income", dtype="number", value=1)
    score = disparate_impact_ratio(privileged=privileged,
                                   unprivileged=unprivileged,
                                   favorable=[favorable],
                                   feature_names=['age', 'race', 'gender', 'income'])
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


def test_average_odds_difference_numpy():
    """Test Average Odds Difference (unbalanced random data, NumPy)"""
    PRIVILEGED_CLASS_GENDER = 1
    UNPRIVILEGED_CLASS_GENDER = 0
    PRIVILEGED_CLASS_RACE = 4
    UNPRIVILEGED_CLASS_RACE = 2

    data_biased = INCOME_DF_BIASED.to_numpy()
    data_unbiased = INCOME_DF_UNBIASED.to_numpy()

    score = average_odds_difference(test=data_biased,
                                    truth=data_unbiased,
                                    privilege_columns=[1, 2],
                                    privilege_values=[PRIVILEGED_CLASS_RACE, PRIVILEGED_CLASS_GENDER],
                                    positive_class=[1],
                                    outputs=[3],
                                    feature_names=['age', 'race', 'gender', 'income'])
    assert score == approx(0.12, abs=0.1)

    score = average_odds_difference(test=data_biased,
                                    truth=data_unbiased,
                                    privilege_columns=[1, 2],
                                    privilege_values=[UNPRIVILEGED_CLASS_RACE, UNPRIVILEGED_CLASS_GENDER],
                                    positive_class=[1],
                                    outputs=[3],
                                    feature_names=['age', 'race', 'gender', 'income'])
    assert score == approx(0.2, abs=0.1)


def test_average_odds_difference_model():
    """Test Average Odds Difference (XGBoost income model)"""
    df = INCOME_DF_BIASED.copy()
    X = df[["age", "race", "gender"]]

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


def test_average_odds_difference_model_numpy():
    """Test Average Odds Difference (XGBoost income model, NumPy)"""
    arr = INCOME_DF_BIASED.to_numpy()
    X = arr[:, 0:3]

    model = Model(XGB_MODEL.predict,
                  feature_names=['age', 'race', 'gender'],
                  output_names=["approved"])

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


def test_average_predictive_value_difference_numpy():
    """Test Average Predictive Value Difference (unbalanced random data, NumPy)"""
    data_biased = INCOME_DF_BIASED.to_numpy()
    data_unbiased = INCOME_DF_UNBIASED.to_numpy()

    PRIVILEGED_CLASS_GENDER = 1
    UNPRIVILEGED_CLASS_GENDER = 0
    PRIVILEGED_CLASS_RACE = 4
    UNPRIVILEGED_CLASS_RACE = 2
    score = average_predictive_value_difference(test=data_biased,
                                                truth=data_unbiased,
                                                privilege_columns=[1, 2],
                                                privilege_values=[PRIVILEGED_CLASS_RACE, PRIVILEGED_CLASS_GENDER],
                                                positive_class=[1],
                                                outputs=[3],
                                                feature_names=['age', 'race', 'gender', 'income'])
    assert score == approx(-0.3, abs=0.1)
    score = average_predictive_value_difference(test=data_biased,
                                                truth=data_unbiased,
                                                privilege_columns=[1, 2],
                                                privilege_values=[UNPRIVILEGED_CLASS_RACE, UNPRIVILEGED_CLASS_GENDER],
                                                positive_class=[1],
                                                outputs=[3],
                                                feature_names=['age', 'race', 'gender', 'income'])
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


def test_average_predictive_value_difference_model_numpy():
    """Test Average Predictive Value Difference (XGB income model, NumPy)"""

    arr = INCOME_DF_BIASED.to_numpy()
    X = arr[:, 0:3]

    model = Model(XGB_MODEL.predict,
                  feature_names=['age', 'race', 'gender'],
                  output_names=["approved"])

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
