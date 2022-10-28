# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys

import pandas as pd
import pytest
import numpy as np

from xai_bench.src import datasets, explainer, metric, experiments, model

LEVEL_1_CONFIG = {
    "rhos": [.5],
    "bench_datasets": ["gaussianPiecewiseConstant"],
    "bench_explainers": ["shap_trustyai","lime_trustyai"],
    "bench_models": ["lr", "dtree", "mlp"],
    "bench_metrics": ["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity",
               "shapley", "shapley_corr", "infidelity"],
    "num_features": 3
}

# LEVEL_2_TESTS = {
#     "rhos": np.linspace(0, 1, 5),
#     "bench_datasets": ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant"],
#     "bench_explainers": ["kernelshaptrustyai"],
#     "bench_models": ["lr", "dtree", "mlp"],
#     "bench_metrics": ["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity",
#                "shapley", "shapley_corr", "infidelity"],
#     "num_features": 5
# }
#
# LEVEL_3_TESTS = {
#     "rhos": np.linspace(0, 1, 5),
#     "bench_datasets": ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant"],
#     "bench_explainers": ["kernelshaptrustyai"],
#     "bench_models": ["lr", "dtree", "mlp"],
#     "bench_metrics": ["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity",
#                "shapley", "shapley_corr", "infidelity"],
#     "num_features": 5
# }

def run_test_config(test_config):
    data = []
    for dataset in test_config['bench_datasets']:
        for rho in test_config['rhos']:
            exp_dataset = datasets.Data(
                name=dataset,
                mode="regression",
                mu=np.zeros(test_config['num_features']),
                rho=rho,
                dim=test_config['num_features'],
                noise=0.01,
                weight=np.arange(test_config['num_features'] - 1, -1, -1),
                num_train_samples=1000,
                num_val_samples=100)
            exp_models = [model.Model(name=m, mode="regression") for m in test_config['bench_models']]
            exp_explainers = [explainer.Explainer(name=e) for e in test_config['bench_explainers']]
            exp_metrics = [metric.Metric(name=m, conditional="observational")
                           for m in test_config['bench_metrics']]
            experiment_results = experiments.Experiment(
                exp_dataset,
                exp_models,
                exp_explainers,
                exp_metrics
            ).get_results()
            data.append(experiment_results)
    return pd.DataFrame(data)


def test_level_1():
    results_df = run_test_config(LEVEL_1_CONFIG)
    results_df.to_pickle("level_1_results.pkl")
