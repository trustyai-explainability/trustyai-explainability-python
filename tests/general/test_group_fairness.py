# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Fairness metrics test suite"""
from typing import List, Optional

from common import *

from pytest import approx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

AIF_DF = pd.read_csv(os.path.join(TEST_DIR, "data/data.csv"))

CREDIT_DF_BIASED = pd.read_csv(os.path.join(TEST_DIR, "data/credit-data-bias-clean.csv"))
CREDIT_BIAS_MODEL = joblib.load(os.path.join(TEST_DIR, "models/credit-bias-model-clean.joblib"))


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


def test_statistical_parity_difference_AIF():
    """Test Statistical Parity Difference (AIF data)"""

    df = AIF_DF.copy()

    privileged = df[df.sex == 1]
    unprivileged = df[df.sex == 0]
    favorable = output("income", dtype="number", value=0)
    score = statistical_parity_difference(privileged=privileged,
                                          unprivileged=unprivileged,
                                          favorable=[favorable])
    assert score == approx(0.19643287553870947, abs=1e-5)


def test_statistical_parity_difference_credit_model():
    """Test Statistical Parity Difference"""
    
    df = CREDIT_DF_BIASED.copy()
    X = df.drop("PaidLoan", axis=1)  
    X = X.astype(float).astype(int)
    
    favorable = output("PaidLoan", dtype="number", value=1)
    model = Model(CREDIT_BIAS_MODEL.predict, dataframe_input=True, output_names=["PaidLoan"])
    
    education_col_idx = X.columns.get_loc("Education")
    privilege_value = int(X["Education"].mode()[0])
    score = statistical_parity_difference_model(X, model, [education_col_idx], [privilege_value], [favorable])
    print(f"Statistical Parity Difference score: {score}")
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


def test_disparate_impact_ratio_AIF():
    """Test Disparate Impact Ratio (AIF data)"""

    df = AIF_DF.copy()

    privileged = df[df.sex == 1]
    unprivileged = df[df.sex == 0]
    favorable = output("income", dtype="number", value=0)
    score = disparate_impact_ratio(privileged=privileged,
                                   unprivileged=unprivileged,
                                   favorable=[favorable])
    assert score == approx(1.28, abs=1e-2)


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


def test_average_odds_difference_credit_model():
    """Test Average Odds Difference """
    
    df = CREDIT_DF_BIASED.copy()
    X = df.drop("PaidLoan", axis=1) 
    X = X.astype(float).astype(int)
    
    model = Model(CREDIT_BIAS_MODEL.predict, dataframe_input=True, output_names=["PaidLoan"])
    
    education_col_idx = X.columns.get_loc("Education")
    privilege_value = int(X["Education"].mode()[0])
    score = average_odds_difference_model(samples=X,  
                                          model=model,
                                          privilege_columns=[education_col_idx],
                                          privilege_values=[privilege_value],
                                          positive_class=[1]) 
    
    print(f"Average Odds Difference score: {score}")
    assert score == approx(0.0, abs=0.09)

def test_average_odds_difference_credit_model_numpy():
    """Test Average Odds Difference """
    
    df = CREDIT_DF_BIASED.copy()
    X_df = df.drop("PaidLoan", axis=1) 
    X_df = X_df.astype(float).astype(int)
    arr = X_df.to_numpy()
    feature_names = X_df.columns.tolist()
    
    model = Model(CREDIT_BIAS_MODEL.predict,
                  feature_names=feature_names,
                  output_names=["PaidLoan"])
    
    education_col_idx = feature_names.index("Education")
    privilege_value = int(X_df["Education"].mode()[0])
    score = average_odds_difference_model(samples=arr,
                                          model=model,
                                          privilege_columns=[education_col_idx],
                                          privilege_values=[privilege_value],
                                          positive_class=[1])
    
    print(f"Average Odds Difference (NumPy) score: {score}")
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


def test_average_predictive_value_difference_credit_model():
    """Test Average Predictive Value Difference """
    
    df = CREDIT_DF_BIASED.copy()
    X = df.drop("PaidLoan", axis=1) 
    X = X.astype(float).astype(int)
    
    model = Model(CREDIT_BIAS_MODEL.predict, dataframe_input=True, output_names=["PaidLoan"])
    
    education_col_idx = X.columns.get_loc("Education")
    privilege_value = int(X["Education"].mode()[0])
    score = average_predictive_value_difference_model(samples=X,
                                                      model=model,
                                                      privilege_columns=[education_col_idx],
                                                      privilege_values=[privilege_value],
                                                      positive_class=[1])
    
    print(f"Average Predictive Value Difference score: {score}")
    assert score == approx(0.0, abs=0.09)

def test_average_predictive_value_difference_credit_model_numpy():
    """Test Average Predictive Value Difference"""
    
    df = CREDIT_DF_BIASED.copy()
    X_df = df.drop("PaidLoan", axis=1)
    X_df = X_df.astype(float).astype(int)
    arr = X_df.to_numpy()
    feature_names = X_df.columns.tolist()
    
    model = Model(CREDIT_BIAS_MODEL.predict,
                  feature_names=feature_names,
                  output_names=["PaidLoan"])
    
    education_col_idx = feature_names.index("Education")
    privilege_value = int(X_df["Education"].mode()[0])
    score = average_predictive_value_difference_model(samples=arr,
                                                      model=model,
                                                      privilege_columns=[education_col_idx],
                                                      privilege_values=[privilege_value],
                                                      positive_class=[1])
    
    print(f"Average Predictive Value Difference (NumPy) score: {score}")
    assert score == approx(0.0, abs=0.09)
