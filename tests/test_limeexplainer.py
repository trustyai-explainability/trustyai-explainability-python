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
    SimplePrediction,
    Type,
    Value,
)
from java.util import Random, ArrayList, List
from trustyai.utils import TestUtils, Config
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
        prediction_output = PredictionOutput(o)
        predictionOutputs.add(prediction_output)
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
    input = PredictionInput([])
    model = TestUtils.getSumSkipModel(0)
    output = (
        model.predictAsync([input])
            .get(Config.INSTANCE.getAsyncTimeout(), Config.INSTANCE.getAsyncTimeUnit())
            .get(0)
    )
    prediction = SimplePrediction(input, output)
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
    features = [FeatureFactory.newNumericalFeature(f"f-num{i}", i) for i in range(4)]

    _input = PredictionInput(features)

    model = PredictionProvider(sumSkipModel)
    output = model.predictAsync([_input]).get().get(0)
    prediction = SimplePrediction(_input, output)
    saliencyMap = limeExplainer.explainAsync(prediction, model).get(
        Config.INSTANCE.getAsyncTimeout(), Config.INSTANCE.getAsyncTimeUnit()
    )
    assert saliencyMap is not None


def testSparseBalance():
    for nf in range(1, 4):
        no_of_samples = 100
        lime_config_no_penalty = LimeConfig() \
            .withPerturbationContext(PerturbationContext(jrandom, DEFAULT_NO_OF_PERTURBATIONS)) \
            .withSamples(no_of_samples) \
            .withPenalizeBalanceSparse(False)
        lime_explainer_no_penalty = LimeExplainer(lime_config_no_penalty)

        features = [mockFeature(i) for i in range(nf)]

        input = PredictionInput(features)
        model = PredictionProvider(sumSkipModel)
        output = model.predictAsync([input]).get().get(0)
        prediction = SimplePrediction(input, output)

        saliency_map_no_penalty = lime_explainer_no_penalty.explainAsync(prediction, model).get()

        assert saliency_map_no_penalty is not None

        decision_name = "sum-but0"
        saliency_no_penalty = saliency_map_no_penalty.get(decision_name)

        lime_config = LimeConfig() \
            .withSamples(no_of_samples) \
            .withPenalizeBalanceSparse(True)
        lime_explainer = LimeExplainer(lime_config)

        saliency_map = lime_explainer.explainAsync(prediction, model).get()
        assert saliency_map is not None

        saliency = saliency_map.get(decision_name)

        for i in range(len(features)):
            score = saliency.getPerFeatureImportance().get(i).getScore()
            score_no_penalty = saliency_no_penalty.getPerFeatureImportance().get(i).getScore()
            assert abs(score) <= abs(score_no_penalty)


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
    output = model.predictAsync([input]).get().get(0)
    prediction = SimplePrediction(input, output)

    saliencyMap = limeExplainer.explainAsync(prediction, model).get()
    assert saliencyMap is not None

    decisionName = "sum-but0"
    saliency = saliencyMap.get(decisionName)
    perFeatureImportance = saliency.getPerFeatureImportance()
    for featureImportance in perFeatureImportance:
        assert -1.0 < featureImportance.getScore() < 1.0
