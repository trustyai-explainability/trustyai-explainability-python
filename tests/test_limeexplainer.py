import sys, os
import pytest

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import trustyai

trustyai.init(path=[
    "./dep/org/kie/kogito/explainability-core/1.5.0.Final/*",
    "./dep/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar",
    "./dep/org/apache/commons/commons-lang3/3.8.1/commons-lang3-3.8.1.jar"
])

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


def sumSkipModel(inputs):
    """SumSkip test model"""
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


def mockFeature(d):
    return FeatureFactory.newNumericalFeature("f-num", d)


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

    model = PredictionProvider(sumSkipModel)
    output = model.predictAsync([_input]).get().get(0)
    prediction = Prediction(_input, output)
    saliencyMap = limeExplainer.explainAsync(prediction, model).get(
        Config.INSTANCE.getAsyncTimeout(), Config.INSTANCE.getAsyncTimeUnit()
    )
    assert saliencyMap is not None


def testSparseBalance():
    for nf in range(1, 4):
        noOfSamples = 100
        limeConfigNoPenalty = LimeConfig() \
            .withPerturbationContext(PerturbationContext(jrandom, DEFAULT_NO_OF_PERTURBATIONS)) \
            .withSamples(noOfSamples) \
            .withPenalizeBalanceSparse(False)
        limeExplainerNoPenalty = LimeExplainer(limeConfigNoPenalty)

        features = ArrayList()
        for i in range(nf):
            features.add(mockFeature(i))

        input = PredictionInput(features)
        model = PredictionProvider(sumSkipModel)
        output = model.predictAsync(toJList([input])).get().get(0)
        prediction = Prediction(input, output)

        saliencyMapNoPenalty = limeExplainerNoPenalty.explainAsync(prediction, model).get()

        assert saliencyMapNoPenalty is not None

        decisionName = "sum-but0"
        saliencyNoPenalty = saliencyMapNoPenalty.get(decisionName);

        limeConfig = LimeConfig() \
            .withSamples(noOfSamples) \
            .withPenalizeBalanceSparse(True)
        limeExplainer = LimeExplainer(limeConfig)

        saliencyMap = limeExplainer.explainAsync(prediction, model).get()
        assert saliencyMap is not None

        saliency = saliencyMap.get(decisionName)

        for i in range(features.size()):
            score = saliency.getPerFeatureImportance().get(i).getScore()
            scoreNoPenalty = saliencyNoPenalty.getPerFeatureImportance().get(i).getScore()
            assert abs(score) <= abs(scoreNoPenalty)


def testNormalizedWeights():
    limeConfig = LimeConfig() \
        .withNormalizeWeights(True) \
        .withPerturbationContext(PerturbationContext(jrandom, 2)) \
        .withSamples(10)
    limeExplainer = LimeExplainer(limeConfig)
    nf = 4
    features = ArrayList()
    for i in range(nf):
        features.add(mockFeature(i))

    input = PredictionInput(features)
    model = PredictionProvider(sumSkipModel)
    output = model.predictAsync(toJList([input])).get().get(0)
    prediction = Prediction(input, output)

    saliencyMap = limeExplainer.explainAsync(prediction, model).get()
    assert saliencyMap is not None

    decisionName = "sum-but0"
    saliency = saliencyMap.get(decisionName)
    perFeatureImportance = saliency.getPerFeatureImportance()
    for featureImportance in perFeatureImportance:
        assert -1.0 < featureImportance.getScore() < 1.0
