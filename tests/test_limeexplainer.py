import sys, os
import pytest
from pytest import approx
import math
import random

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import trustyai

trustyai.init()

DEFAULT_NO_OF_PERTURBATIONS = 1

from trustyai.local.lime import LimeConfig, LimeExplainer
from trustyai.model import (
    PerturbationContext,
    PredictionInput,
    Prediction,
    FeatureFactory,
)
from java.util import Random, ArrayList, List
from trustyai.utils import TestUtils, Config
from org.kie.kogito.explainability.local import (
    LocalExplanationException,
)

jrandom = Random()
jrandom.setSeed(0)


def testEmptyPrediction():
    limeConfig = (
        LimeConfig()
        .withPerturbationContext(
            PerturbationContext(jrandom, DEFAULT_NO_OF_PERTURBATIONS)
        )
        .withSamples(10)
    )
    limeExplainer = LimeExplainer(limeConfig)
    input = PredictionInput(ArrayList())
    model = TestUtils.getSumSkipModel(0)
    output = (
        model.predictAsync(List.of(input))
        .get(Config.INSTANCE.getAsyncTimeout(), Config.INSTANCE.getAsyncTimeUnit())
        .get(0)
    )
    prediction = Prediction(input, output)
    with pytest.raises(LocalExplanationException):
        limeExplainer.explainAsync(prediction, model)


def testNonEmptyInput():
    limeConfig = (
        LimeConfig()
        .withPerturbationContext(
            PerturbationContext(jrandom, DEFAULT_NO_OF_PERTURBATIONS)
        )
        .withSamples(10)
    )
    limeExplainer = LimeExplainer(limeConfig)
    features = ArrayList()
    for i in range(4):
        features.add(FeatureFactory.newNumericalFeature(f"f-num{i}", i))
    input = PredictionInput(features)
    model = TestUtils.getSumSkipModel(0)
    output = (
        model.predictAsync(List.of(input))
        .get(Config.INSTANCE.getAsyncTimeout(), Config.INSTANCE.getAsyncTimeUnit())
        .get(0)
    )
    prediction = Prediction(input, output)
    saliencyMap = limeExplainer.explainAsync(prediction, model).get(
        Config.INSTANCE.getAsyncTimeout(), Config.INSTANCE.getAsyncTimeUnit()
    )

    assert saliencyMap is not None
