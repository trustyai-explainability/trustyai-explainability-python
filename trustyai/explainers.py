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
from org.optaplanner.core.config.solver.termination import TerminationConfig
from java.lang import Long

SolverConfigBuilder = _SolverConfigBuilder
CounterfactualConfig = _CounterfactualConfig
LimeConfig = _LimeConfig


class CounterfactualExplainer:
    """Wrapper for TrustyAI's counterfactual explainer"""

    def __init__(self, steps=10_000) -> None:
        self._termination_config = TerminationConfig().withScoreCalculationCountLimit(
            Long.valueOf(steps)
        )
        self._solver_config = (
            SolverConfigBuilder.builder()
            .withTerminationConfig(self._termination_config)
            .build()
        )
        self._cf_config = CounterfactualConfig().withSolverConfig(self._solver_config)

        self._explainer = _CounterfactualExplainer(self._cf_config)

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
