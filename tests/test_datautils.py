import sys, os
import pytest
from pytest import approx
import math

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import trustyai

trustyai.init()

from trustyai.utils import DataUtils
from trustyai.model import PerturbationContext
from java.util import Random

jrandom = Random()


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


def test_HammingDistanceDouble():
    x = [2, 1]
    y = [2, 3]
    distance = DataUtils.hammingDistance(x, y)
    assert distance == approx(1, 1e-1)


def test_HammingDistanceString():
    x = "test1"
    y = "test2"
    distance = DataUtils.hammingDistance(x, y)
    assert distance == approx(1, 1e-1)


def test_DoublesToFeatures():
    inputs = [1 if i % 2 == 0 else 0 for i in range(10)]
    features = DataUtils.doublesToFeatures(inputs)
    assert features is not None
    assert len(features) == 10
    for f in features:
        assert f is not None
        assert f.getName() is not None
        assert f.getValue() is not None


def test_ExponentialSmoothingKernel():
    x = 0.218
    k = DataUtils.exponentialSmoothingKernel(x, 2)
    assert k == approx(0.994, 1e-3)


def test_PerturbFeaturesEmpty():
    features = []
    perturbationContext = PerturbationContext(jrandom, 0)
    newFeatures = DataUtils.perturbFeatures(features, perturbationContext)
    assert newFeatures is not None
    assert len(features) == newFeatures.size()
