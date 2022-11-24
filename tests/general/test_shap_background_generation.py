"""SHAP background generation  test suite"""

import pytest
import numpy as np
import math

from trustyai.explainers.shap import BackgroundGenerator
from trustyai.model import Model
from trustyai.utils.data_conversions import prediction_object_to_numpy


def test_random_generation():
    """Test that random sampling recovers samples from distribution"""
    seed = 0
    np.random.seed(seed)
    data = np.random.rand(100, 5)
    background_ta = BackgroundGenerator(data).sample(5)
    background = prediction_object_to_numpy(background_ta)

    assert len(background) == 5
    for row in background:
        assert row in data


def test_kmeans_generation():
    """Test that k-means recovers centroids of well-clustered data"""

    seed = 0
    clusters = 5
    np.random.seed(seed)

    data = []
    ground_truth = []
    for cluster in range(clusters):
        data.append(np.random.rand(100 // clusters, 5) + cluster * 10)
        ground_truth.append(np.array([cluster * 10] * 5))
    data = np.vstack(data)
    ground_truth = np.vstack(ground_truth)
    background_ta = BackgroundGenerator(data).kmeans(clusters)
    background = prediction_object_to_numpy(background_ta)

    assert len(background) == 5
    for row in background:
        ground_truth_idx = math.floor(row[0] / 10)
        assert np.linalg.norm(row - ground_truth[ground_truth_idx]) < 2.5


def test_counterfactual_generation():
    """Test that k-means recovers centroids of well-clustered data"""

    seed = 0
    np.random.seed(seed)
    data = np.random.rand(100, 5)

    model = Model(lambda x: x.sum(1), arrow=False)
    goal = np.array([5])

    background_ta = BackgroundGenerator(data).counterfactual(goal, model, 100)
    background = prediction_object_to_numpy(background_ta)

    for row in background:
        print(row)
        assert np.linalg.norm(5 - model(row)) < .01