"""Explainers module"""
# pylint: disable = import-error, too-few-public-methods
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display
import pandas as pd
import numpy as np

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
    ShapKernelExplainer as _ShapKernelExplainer,
)

from org.kie.kogito.explainability.model import (
    CounterfactualPrediction,
    EncodingParams,
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
        for feature_importance in self._saliencies.get(
            decision
        ).getPerFeatureImportance():
            result += f"\t{feature_importance.getFeature().name}: {feature_importance.getScore()}\n"
        return result

    def map(self):
        """Return saliencies map"""
        return self._saliencies

    def plot(self, decision: str):
        """Plot saliencies"""
        dictionary = {}
        for feature_importance in self._saliencies.get(
            decision
        ).getPerFeatureImportance():
            dictionary[
                feature_importance.getFeature().name
            ] = feature_importance.getScore()

        colours = ["r" if i < 0 else "g" for i in dictionary.values()]
        plt.title(f"LIME explanation for '{decision}'")
        plt.barh(
            range(len(dictionary)), dictionary.values(), align="center", color=colours
        )
        plt.yticks(range(len(dictionary)), list(dictionary.keys()))
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
            .withEncodingParams(EncodingParams(0.07, 0.3))
            .withAdaptiveVariance(True)
            .withPenalizeBalanceSparse(penalise_sparse_balance)
        )

        self._explainer = _LimeExplainer(self._lime_config)

    def explain(self, prediction, model: PredictionProvider) -> LimeExplanation:
        """Request for a LIME explanation given a prediction and a model"""
        return LimeExplanation(self._explainer.explainAsync(prediction, model).get())


# pylint: disable=invalid-name
class SHAPResults:
    """Wrapper for TrustyAI's SHAPResults object"""

    def __init__(self, shap_results, background):
        self.shap_results = shap_results
        self.background = background

    def getSaliencies(self):
        """Wrapper for ShapResults.getSaliencies()"""
        return self.shap_results.getSaliencies()

    def getFnull(self):
        """Wrapper for ShapResults.getFnull()"""
        return self.shap_results.getFnull()

    def visualize_as_dataframe(self):
        """Print out the SHAP values as a formatted dataframe"""

        def _color_feature_values(feature_values, background_vals):
            """Internal function for the dataframe visualization"""
            formats = []
            for i, feature_value in enumerate(feature_values[1:-1]):
                if feature_value < background_vals[i]:
                    formats.append("background-color:#ee0000")
                elif feature_value > background_vals[i]:
                    formats.append("background-color:#13ba3c")
                else:
                    formats.append(None)
            return [None] + formats + [None]

        for i, saliency in enumerate(self.shap_results.getSaliencies()):
            background_mean_feature_values = np.mean(
                [
                    [f.getValue().asNumber() for f in pi.getFeatures()]
                    for pi in self.background
                ],
                0,
            ).tolist()
            feature_values = [
                pfi.getFeature().getValue().asNumber()
                for pfi in saliency.getPerFeatureImportance()
            ]
            shap_values = [pfi.getScore() for pfi in saliency.getPerFeatureImportance()]
            feature_names = [
                str(pfi.getFeature().getName())
                for pfi in saliency.getPerFeatureImportance()
            ]
            columns = ["Mean Background Value", "Feature Value", "SHAP Value"]
            visualizer_data_frame = pd.DataFrame(
                [background_mean_feature_values, feature_values, shap_values],
                index=columns,
                columns=feature_names,
            ).T
            fnull = self.shap_results.getFnull().getEntry(i)

            visualizer_data_frame = pd.concat(
                [
                    pd.DataFrame(
                        [["-", "-", fnull]], index=["Background"], columns=columns
                    ),
                    visualizer_data_frame,
                    pd.DataFrame(
                        [[fnull, sum(shap_values) + fnull, sum(shap_values) + fnull]],
                        index=["Prediction"],
                        columns=columns,
                    ),
                ]
            )
            style = visualizer_data_frame.style.background_gradient(
                LinearSegmentedColormap.from_list(
                    name="rwg", colors=["#ee0000", "#ffffff", "#13ba3c"]
                ),
                subset=(slice(feature_names[0], feature_names[-1]), "SHAP Value"),
                vmin=-1 * max(np.abs(shap_values)),
                vmax=max(np.abs(shap_values)),
            )
            style.set_caption(f"Explanation of {saliency.getOutput().getName()}")
            display(
                style.apply(
                    _color_feature_values,
                    background_vals=background_mean_feature_values,
                    subset="Feature Value",
                    axis=0,
                )
            )

    def visualize_as_candlestick_plot(self):
        """Plot each SHAP explanation as a candlestick plot"""
        plt.style.use(
            "https://raw.githubusercontent.com/RobGeada/stylelibs/main/material_rh.mplstyle"
        )

        for i, saliency in enumerate(self.shap_results.getSaliencies()):
            shap_values = [pfi.getScore() for pfi in saliency.getPerFeatureImportance()]
            feature_names = [
                str(pfi.getFeature().getName())
                for pfi in saliency.getPerFeatureImportance()
            ]
            fnull = self.shap_results.getFnull().getEntry(i)
            prediction = fnull + sum(shap_values)
            plt.figure()
            pos = fnull
            for j, shap_value in enumerate(shap_values):
                color = "#ee0000" if shap_value < 0 else "#13ba3c"
                width = 0.9
                if j > 0:
                    plt.plot([j - 0.5, j + width / 2 * 0.99], [pos, pos], color=color)
                plt.bar(j, height=shap_value, bottom=pos, color=color, width=width)
                pos += shap_values[j]

                if j != len(shap_values) - 1:
                    plt.plot([j - width / 2 * 0.99, j + 0.5], [pos, pos], color=color)

            plt.axhline(
                fnull,
                color="#444444",
                linestyle="--",
                zorder=0,
                label="Background Value",
            )
            plt.axhline(prediction, color="#444444", zorder=0, label="Prediction")
            plt.legend()

            ticksize = np.diff(plt.gca().get_yticks())[0]
            plt.ylim(
                plt.gca().get_ylim()[0] - ticksize / 2,
                plt.gca().get_ylim()[1] + ticksize / 2,
            )
            plt.xticks(np.arange(len(feature_names)), feature_names)
            plt.ylabel(saliency.getOutput().getName())
            plt.xlabel("Feature SHAP Value")
            plt.title(f"Explanation of {saliency.getOutput().getName()}")
            plt.show()


class SHAPExplainer:
    """Wrapper for TrustyAI's SHAP explainer"""

    def __init__(
        self,
        background: List[_PredictionInput],
        samples=None,
        batch_size=20,
        seed=0,
        perturbations=0,
        link_type: Optional[_ShapConfig.LinkType] = None,
    ):
        if not link_type:
            link_type = _ShapConfig.LinkType.IDENTITY
        self._jrandom = Random()
        self._jrandom.setSeed(seed)
        self.background = background
        perturbation_context = PerturbationContext(self._jrandom, perturbations)
        self._configbuilder = (
            _ShapConfig.builder()
            .withLink(link_type)
            .withBatchSize(batch_size)
            .withPC(perturbation_context)
            .withBackground(background)
        )
        if samples is not None:
            self._configbuilder.withNSamples(JInt(samples))
        self._config = self._configbuilder.build()
        self._explainer = _ShapKernelExplainer(self._config)

    def explain(self, prediction, model: PredictionProvider) -> SHAPResults:
        """Request for a SHAP explanation given a prediction and a model"""
        return SHAPResults(
            self._explainer.explainAsync(prediction, model).get(), self.background
        )
