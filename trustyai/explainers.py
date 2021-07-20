"""Explainers module"""
# pylint: disable = import-error, too-few-public-methods
from typing import Dict
from org.kie.kogito.explainability.local.counterfactual import (
    CounterfactualExplainer as _CounterfactualExplainer,
    CounterfactualConfigurationFactory as _CounterfactualConfigurationFactory,
    CounterfactualResult,
)
from org.kie.kogito.explainability.local.lime import (
    LimeConfig as _LimeConfig,
    LimeExplainer as _LimeExplainer,
)
from org.optaplanner.core.config.solver import SolverConfig
from org.kie.kogito.explainability.model import Prediction, PredictionProvider, Saliency


CounterfactualConfigurationFactory = _CounterfactualConfigurationFactory
LimeConfig = _LimeConfig


class CounterfactualExplainer:
    """Wrapper for TrustyAI's counterfactual explainer"""

    def __init__(self, config: SolverConfig) -> None:
        self._explainer = (
            _CounterfactualExplainer.builder().withSolverConfig(config).build()
        )

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
