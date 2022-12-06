"""Explainers module"""
# pylint: disable=duplicate-code
from .counterfactuals import CounterfactualResult, CounterfactualExplainer
from .lime import LimeExplainer, LimeResults
from .shap import SHAPExplainer, SHAPResults, BackgroundGenerator
from jpype import (
    JImplements,
    JOverride,
)

@JImplements("org.kie.trustyai.explainability.local.LocalExplainer", deferred=True)
class LocalExplainer:

    def __init__(self, wrapped):
        self.wrapped = wrapped

    @JOverride
    def explainAsync(self, prediction, model):
        return self.wrapped.explainAsync(prediction, model)

