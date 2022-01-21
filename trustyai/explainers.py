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

from org.kie.kogito.explainability.model import (
    CounterfactualPrediction,
    PredictionProvider,
    Saliency,
    PerturbationContext,
)
from org.optaplanner.core.config.solver.termination import TerminationConfig
from java.lang import Long
from java.util import Random

SolverConfigBuilder = _SolverConfigBuilder
CounterfactualConfig = _CounterfactualConfig
LimeConfig = _LimeConfig


class CounterfactualExplainer:
    """Wrapper for TrustyAI's counterfactual explainer"""

    def __init__(self, steps=10_000):
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
        self, prediction: CounterfactualPrediction, model: PredictionProvider
    ) -> CounterfactualResult:
        """Request for a counterfactual explanation given a prediction and a model"""
        return self._explainer.explainAsync(prediction, model).get()


# pylint: disable=too-many-arguments
class LimeExplainer:
    """Wrapper for TrustyAI's LIME explainer"""

    def __init__(
        self,
        perturbations=1,
        seed=0,
        samples=10,
        penalise_sparse_balance=True,
        normalise_weights=True,
    ):
        # build LIME configuration
        self._jrandom = Random()
        self._jrandom.setSeed(seed)

        self._lime_config = (
            LimeConfig()
            .withNormalizeWeights(normalise_weights)
            .withPerturbationContext(PerturbationContext(self._jrandom, perturbations))
            .withSamples(samples)
            .withPenalizeBalanceSparse(penalise_sparse_balance)
        )

        self._explainer = _LimeExplainer(self._lime_config)

    def explain(self, prediction, model: PredictionProvider) -> Dict[str, Saliency]:
        """Request for a LIME explanation given a prediction and a model"""
        return self._explainer.explainAsync(prediction, model).get()
