# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys
import pytest
import time
import numpy as np

from benchmark_common import mean_impact_score
import test_counterfactualexplainer as tcf
import test_limeexplainer as tlime

from trustyai.explainers import LimeExplainer, SHAPExplainer
from trustyai.model import feature, PredictionInput
from trustyai.utils import TestUtils

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../general/")


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
def test_lime_impact_score(benchmark):
    benchmark.extra_info['impact_score'] = tlime.test_impact_score()
    benchmark(tlime.test_impact_score)


@pytest.mark.benchmark(
    group="lime", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_sumskip_lime_impact_score(benchmark):
    np.random.seed(0)
    explainer = LimeExplainer()
    model = TestUtils.getSumSkipModel(0)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-10, 10), dtype="number") for i in range(4)])
    benchmark.extra_info['impact_score'] = mean_impact_score(explainer, model, data)
    benchmark(mean_impact_score, explainer, model, data)


@pytest.mark.benchmark(
    group="shap", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_sumskip_shap_impact_score(benchmark):
    np.random.seed(0)
    background = []
    for i in range(10):
        background.append(PredictionInput([feature(name=f"f-num{i}", value=np.random.randint(-10, 10), dtype="number") for i in range(4)]))
    explainer = SHAPExplainer(background)
    model = TestUtils.getSumSkipModel(0)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-10, 10), dtype="number") for i in range(4)])
    benchmark.extra_info['impact_score'] = mean_impact_score(explainer, model, data)
    benchmark(mean_impact_score, explainer, model, data)


@pytest.mark.benchmark(
    group="lime", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_sumthreshold_lime_impact_score(benchmark):
    np.random.seed(0)
    explainer = LimeExplainer()
    center = 100.0
    epsilon = 10.0
    model = TestUtils.getSumThresholdModel(center, epsilon)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(4)])
    benchmark.extra_info['impact_score'] = mean_impact_score(explainer, model, data)
    benchmark(mean_impact_score, explainer, model, data)


@pytest.mark.benchmark(
    group="shap", min_rounds=10, timer=time.time, disable_gc=True, warmup=True
)
def test_sumthreshold_shap_impact_score(benchmark):
    np.random.seed(0)
    background = []
    for i in range(100):
        background.append(PredictionInput([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(4)]))
    explainer = SHAPExplainer(background)
    center = 100.0
    epsilon = 10.0
    model = TestUtils.getSumThresholdModel(center, epsilon)
    data = []
    for i in range(100):
        data.append([feature(name=f"f-num{i}", value=np.random.randint(-100, 100), dtype="number") for i in range(4)])
    benchmark.extra_info['impact_score'] = mean_impact_score(explainer, model, data)
    benchmark(mean_impact_score, explainer, model, data)
