# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Data utils test suite"""
import sys
import os
from pytest import approx
import random

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import trustyai

trustyai.init()

from trustyai.utils import DataUtils
from trustyai.model import PerturbationContext, FeatureFactory
from java.util import Random

jrandom = Random()


def test_get_mean():
    """Test GetMean"""
    data = [2, 4, 3, 5, 1]
    assert DataUtils.getMean(data) == approx(3, 1e-6)


def test_get_std_dev():
    """Test GetStdDev"""
    data = [2, 4, 3, 5, 1]
    assert DataUtils.getStdDev(data, 3) == approx(1.41, 1e-2)


def test_gaussian_kernel():
    """Test Gaussian Kernel"""
    x = 0.0
    k = DataUtils.gaussianKernel(x, 0, 1)
    assert k == approx(0.398, 1e-2)
    x = 0.218
    k = DataUtils.gaussianKernel(x, 0, 1)
    assert k == approx(0.389, 1e-2)


def test_euclidean_distance():
    """Test Euclidean distance"""
    x = [1, 1]
    y = [2, 3]
    distance = DataUtils.euclideanDistance(x, y)
    assert approx(distance, 1e-3) == 2.236


def test_hamming_distance_double():
    """Test Hamming distance for doubles"""
    x = [2, 1]
    y = [2, 3]
    distance = DataUtils.hammingDistance(x, y)
    assert distance == approx(1, 1e-1)


def test_hamming_distance_string():
    """Test Hamming distance for strings"""
    x = "test1"
    y = "test2"
    distance = DataUtils.hammingDistance(x, y)
    assert distance == approx(1, 1e-1)


def test_doubles_to_features():
    """Test doubles to features"""
    inputs = [1 if i % 2 == 0 else 0 for i in range(10)]
    features = DataUtils.doublesToFeatures(inputs)
    assert features is not None
    assert len(features) == 10
    for f in features:
        assert f is not None
        assert f.getName() is not None
        assert f.getValue() is not None


def test_exponential_smoothing_kernel():
    """Test exponential smoothing kernel"""
    x = 0.218
    k = DataUtils.exponentialSmoothingKernel(x, 2)
    assert k == approx(0.994, 1e-3)


def test_perturb_features_empty():
    """Test perturb empty features"""
    features = []
    perturbationContext = PerturbationContext(jrandom, 0)
    newFeatures = DataUtils.perturbFeatures(features, perturbationContext)
    assert newFeatures is not None
    assert len(features) == newFeatures.size()


def test_random_distribution_generation():
    """Test random distribution generation"""
    dataDistribution = DataUtils.generateRandomDataDistribution(10, 10, jrandom)
    assert dataDistribution is not None
    assert dataDistribution.asFeatureDistributions() is not None
    for featureDistribution in dataDistribution.asFeatureDistributions():
        assert featureDistribution is not None


def test_linearized_numeric_features():
    """Test linearised numeric features"""
    f = FeatureFactory.newNumericalFeature("f-num", 1.0)
    features = [f]
    linearizedFeatures = DataUtils.getLinearizedFeatures(features)
    assert len(features) == linearizedFeatures.size()


def test_sample_with_replacement():
    """Test sample with replacement"""
    emptyValues = []
    emptySamples = DataUtils.sampleWithReplacement(emptyValues, 1, jrandom)
    assert emptySamples is not None
    assert emptySamples.size() == 0

    values = DataUtils.generateData(0, 1, 100, jrandom)
    sampleSize = 10
    samples = DataUtils.sampleWithReplacement(values, sampleSize, jrandom)
    assert samples is not None
    assert samples.size() == sampleSize
    assert samples[random.randint(0, sampleSize - 1)] in values

    largerSampleSize = 300
    largerSamples = DataUtils.sampleWithReplacement(values, largerSampleSize, jrandom)
    assert largerSamples is not None
    assert largerSampleSize == largerSamples.size()
    assert largerSamples[random.randint(0, largerSampleSize - 1)] in largerSamples
