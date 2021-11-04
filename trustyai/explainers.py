"""Explainers module"""
# pylint: disable = import-error, too-few-public-methods
from typing import Dict
from org.kie.kogito.explainability.local.counterfactual import (
    CounterfactualExplainer as _CounterfactualExplainer,
    CounterfactualResult,
    SolverConfigBuilder as _SolverConfigBuilder,
    CounterfactualConfig as _CounterfactualConfig,
)
from org.kie.kogito.explainability.local.lime import (
    LimeConfig as _LimeConfig,
    LimeExplainer as _LimeExplainer,
)

from org.kie.kogito.explainability.model import Prediction, PredictionProvider, Saliency

SolverConfigBuilder = _SolverConfigBuilder
CounterfactualConfig = _CounterfactualConfig
LimeConfig = _LimeConfig


class CounterfactualExplainer:
    """Wrapper for TrustyAI's counterfactual explainer"""

    def __init__(self, config: CounterfactualConfig) -> None:
        self._explainer = _CounterfactualExplainer(config)

    def explain(
        self, prediction: Prediction, model: PredictionProvider
    ) -> CounterfactualResult:
        """Request for a counterfactual explanation given a prediction and a model"""
        return self._explainer.explainAsync(prediction, model).get()


class LimeExplainer:
    """Wrapper for TrustyAI's LIME explainer"""

    def __init__(self, config: LimeConfig):
        self._explainer = _LimeExplainer(config)

    def explain(self, prediction, model: PredictionProvider) -> Dict[str, Saliency]:
        """Request for a LIME explanation given a prediction and a model"""
        return self._explainer.explainAsync(prediction, model).get()
