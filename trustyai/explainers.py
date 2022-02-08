"""Explainers module"""
# pylint: disable = import-error, too-few-public-methods
from typing import Dict, Optional, List
import matplotlib.pyplot as plt

from jpype import JInt
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

from org.kie.kogito.explainability.local.shap import (
    ShapConfig as _ShapConfig,
    ShapResults,
    ShapKernelExplainer as _ShapKernelExplainer,
)

from org.kie.kogito.explainability.model import (
    CounterfactualPrediction,
    PredictionProvider,
    Saliency,
    PerturbationContext,
    PredictionInput as _PredictionInput,
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


class LimeExplanation:
    """Encapsulate LIME results"""

    def __init__(self, saliencies: Dict[str, Saliency]):
        self._saliencies = saliencies

    def show(self, decision: str) -> str:
        """Return saliencies for a decision"""
        result = f"Saliencies for '{decision}':\n"
        for f in self._saliencies.get(decision).getPerFeatureImportance():
            result += f'\t{f.getFeature().name}: {f.getScore()}\n'
        return result

    def map(self):
        return self._saliencies

    def plot(self, decision: str):
        d = {}
        for f in self._saliencies.get(decision).getPerFeatureImportance():
            d[f.getFeature().name] = f.getScore()

        colours = ['r' if i < 0 else 'g' for i in d.values()]
        plt.title(f"LIME explanation for '{decision}'")
        plt.barh(range(len(d)), d.values(), align='center', color=colours)
        plt.yticks(range(len(d)), list(d.keys()))
        plt.tight_layout()


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

    def explain(self, prediction, model: PredictionProvider) -> LimeExplanation:
        """Request for a LIME explanation given a prediction and a model"""
        return LimeExplanation(self._explainer.explainAsync(prediction, model).get())


class SHAPExplainer:
    """Wrapper for TrustyAI's SHAP explainer"""

    def __init__(
            self,
            background: List[_PredictionInput],
            samples=100,
            seed=0,
            perturbations=0,
            link_type: Optional[_ShapConfig.LinkType] = None,
    ):
        if not link_type:
            link_type = _ShapConfig.LinkType.IDENTITY
        self._jrandom = Random()
        self._jrandom.setSeed(seed)
        perturbation_context = PerturbationContext(self._jrandom, perturbations)
        self._config = (
            _ShapConfig.builder()
                .withLink(link_type)
                .withPC(perturbation_context)
                .withBackground(background)
                .withNSamples(JInt(samples))
                .build()
        )
        self._explainer = _ShapKernelExplainer(self._config)

    def explain(self, prediction, model: PredictionProvider) -> List[ShapResults]:
        """Request for a SHAP explanation given a prediction and a model"""
        return self._explainer.explainAsync(prediction, model).get()
