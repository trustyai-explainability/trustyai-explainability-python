# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys
import pytest
import time
import numpy as np

from trustyai.explainers import LimeExplainer, SHAPExplainer
from trustyai.model import feature, PredictionInput
from trustyai.utils import TestModels
from trustyai.metrics.saliency import mean_impact_score, classification_fidelity, local_saliency_f1

from org.kie.trustyai.explainability.model import (
    PredictionInputsDataDistribution,
)

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../general/")

import test_counterfactualexplainer as tcf

@pytest.mark.benchmark(
    group="counterfactuals", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_counterfactual_match(benchmark):
    """Counterfactual match"""
    benchmark(tcf.test_counterfactual_match)


@pytest.mark.benchmark(
    group="counterfactuals", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_non_empty_input(benchmark):
    """Counterfactual non-empty input"""
    benchmark(tcf.test_non_empty_input)


@pytest.mark.benchmark(
    group="counterfactuals", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_counterfactual_match_python_model(benchmark):
    """Counterfactual match (Python model)"""
    benchmark(tcf.test_counterfactual_match_python_model)


@pytest.mark.benchmark(
    group="lime", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_sumskip_lime_impact_score_at_2(benchmark):
    no_of_features = 10
    np.random.seed(0)
    explainer = LimeExplainer()
    model = TestModels.getSumSkipModel(0)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-10, 10), dtype="number") for i in range(no_of_features)])
    benchmark.extra_info['metric'] = mean_impact_score(explainer, model, data)
    benchmark(mean_impact_score, explainer, model, data)


@pytest.mark.benchmark(
    group="shap", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_sumskip_shap_impact_score_at_2(benchmark):
    no_of_features = 10
    np.random.seed(0)
    background = []
    for i in range(10):
        background.append(PredictionInput([feature(name=f"f-num{i}", value=np.random.randint(-10, 10), dtype="number") for i in range(no_of_features)]))
    explainer = SHAPExplainer(background, samples=10000)
    model = TestModels.getSumSkipModel(0)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-10, 10), dtype="number") for i in range(no_of_features)])
    benchmark.extra_info['metric'] = mean_impact_score(explainer, model, data)
    benchmark(mean_impact_score, explainer, model, data)


@pytest.mark.benchmark(
    group="lime", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_sumthreshold_lime_impact_score_at_2(benchmark):
    no_of_features = 10
    np.random.seed(0)
    explainer = LimeExplainer()
    center = 100.0
    epsilon = 10.0
    model = TestModels.getSumThresholdModel(center, epsilon)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(no_of_features)])
    benchmark.extra_info['metric'] = mean_impact_score(explainer, model, data)
    benchmark(mean_impact_score, explainer, model, data)


@pytest.mark.benchmark(
    group="shap", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_sumthreshold_shap_impact_score_at_2(benchmark):
    no_of_features = 10
    np.random.seed(0)
    background = []
    for i in range(100):
        background.append(PredictionInput([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(no_of_features)]))
    explainer = SHAPExplainer(background, samples=10000)
    center = 100.0
    epsilon = 10.0
    model = TestModels.getSumThresholdModel(center, epsilon)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(no_of_features)])
    benchmark.extra_info['metric'] = mean_impact_score(explainer, model, data)
    benchmark(mean_impact_score, explainer, model, data)


@pytest.mark.benchmark(
    group="lime", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_lime_fidelity(benchmark):
    no_of_features = 10
    np.random.seed(0)
    explainer = LimeExplainer()
    model = TestModels.getEvenSumModel(0)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(no_of_features)])
    benchmark.extra_info['metric'] = classification_fidelity(explainer, model, data)
    benchmark(classification_fidelity, explainer, model, data)


@pytest.mark.benchmark(
    group="shap", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_shap_fidelity(benchmark):
    no_of_features = 10
    np.random.seed(0)
    background = []
    for i in range(10):
        background.append(PredictionInput(
            [feature(name=f"f-num{i}", value=np.random.randint(-10, 10), dtype="number") for i in
             range(no_of_features)]))
    explainer = SHAPExplainer(background, samples=10000)
    model = TestModels.getEvenSumModel(0)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in
                     range(no_of_features)])
    benchmark.extra_info['metric'] = classification_fidelity(explainer, model, data)
    benchmark(classification_fidelity, explainer, model, data)


@pytest.mark.benchmark(
    group="lime", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_lime_local_saliency_f1(benchmark):
    no_of_features = 10
    np.random.seed(0)
    explainer = LimeExplainer()
    model = TestModels.getEvenSumModel(0)
    output_name = "sum-even-but0"
    data = []
    for i in range(100):
        data.append(PredictionInput([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(no_of_features)]))
    distribution = PredictionInputsDataDistribution(data)
    benchmark.extra_info['metric'] = local_saliency_f1(output_name, model, explainer, distribution, 2, 10)
    benchmark(local_saliency_f1, output_name, model, explainer, distribution, 2, 10)


@pytest.mark.benchmark(
    group="shap", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_shap_local_saliency_f1(benchmark):
    no_of_features = 10
    np.random.seed(0)
    background = []
    for i in range(10):
        background.append(PredictionInput(
            [feature(name=f"f-num{i}", value=np.random.randint(-10, 10), dtype="number") for i in
             range(no_of_features)]))
    explainer = SHAPExplainer(background, samples=10000)
    model = TestModels.getEvenSumModel(0)
    output_name = "sum-even-but0"
    data = []
    for i in range(100):
        data.append(PredictionInput([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(no_of_features)]))
    distribution = PredictionInputsDataDistribution(data)
    benchmark.extra_info['metric'] = local_saliency_f1(output_name, model, explainer, distribution, 2, 10)
    benchmark(local_saliency_f1, output_name, model, explainer, distribution, 2, 10)