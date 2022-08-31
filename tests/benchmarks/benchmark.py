# pylint: disable=R0801
"""Common methods and models for tests"""
import os
import sys
import pytest
import time

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../general/")

import test_counterfactualexplainer as tcf

@pytest.mark.benchmark(
    group="counterfactuals",
    min_rounds=5,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_counterfactual_match(benchmark):
    """Counterfactual match"""
    benchmark(tcf.test_counterfactual_match)
