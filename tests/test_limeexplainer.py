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
    PredictionProvider,
    Output,
    PredictionOutput,
    Type,
    Value,
)
from java.util import Random, ArrayList, List
from trustyai.utils import TestUtils, Config, toJList
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
    features = toJList(
        [FeatureFactory.newNumericalFeature(f"f-num{i}", i) for i in range(4)]
    )
    _input = PredictionInput(features)

    def sumSkipModel(inputs):
        predictionOutputs = ArrayList()
        for predictionInput in inputs:
            features = predictionInput.getFeatures()
            result = 0.0
            for i in range(features.size()):
                if i != 0:
                    result += features.get(i).getValue().asNumber()
            o = [Output(f"sum-but0", Type.NUMBER, Value(result), 1.0)]
            predictionOutput = PredictionOutput(toJList(o))
            predictionOutputs.add(predictionOutput)
        return predictionOutputs

    model = PredictionProvider(sumSkipModel)
    output = model.predictAsync([_input]).get().get(0)
    prediction = Prediction(_input, output)
    saliencyMap = limeExplainer.explainAsync(prediction, model).get(
        Config.INSTANCE.getAsyncTimeout(), Config.INSTANCE.getAsyncTimeUnit()
    )
    print(saliencyMap)
    assert saliencyMap is not None
