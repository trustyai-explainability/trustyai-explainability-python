import sys, os
import pytest
from pytest import approx
import math

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import trustyai

trustyai.init()
from trustyai import DataUtils


def test_GetMean():
    data = [2, 4, 3, 5, 1]
    assert DataUtils.getMean(data) == approx(3, 1e-6)


def test_GetStdDev():
    data = [2, 4, 3, 5, 1]
    assert DataUtils.getStdDev(data, 3) == approx(1.41, 1e-2)


def test_GaussianKernel():
    x = 0.0
    k = DataUtils.gaussianKernel(x, 0, 1)
    assert k == approx(0.398, 1e-2)
    x = 0.218
    k = DataUtils.gaussianKernel(x, 0, 1)
    assert k == approx(0.389, 1e-2)


def test_EuclideanDistance():
    x = [1, 1]
    y = [2, 3]
    distance = DataUtils.euclideanDistance(x, y)
    assert 2.236 == approx(distance, 1e-3)
