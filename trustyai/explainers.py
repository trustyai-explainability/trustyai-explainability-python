"""Explainers module"""
# pylint: disable = import-error, too-few-public-methods
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import pandas.io.formats.style as Styler
import numpy as np
from jpype import JInt
from org.kie.kogito.explainability.local.counterfactual import (
    CounterfactualExplainer as _CounterfactualExplainer,
    CounterfactualResult as _CounterfactualResult,
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
from trustyai.utils._visualisation import (
    ExplanationVisualiser,
    DEFAULT_STYLE as ds,
    DEFAULT_RC_PARAMS as drcp,
)

SolverConfigBuilder = _SolverConfigBuilder
CounterfactualConfig = _CounterfactualConfig
LimeConfig = _LimeConfig


class CounterfactualResult(ExplanationVisualiser):
    """Encapsulate counterfactual results"""

    def __init__(self, result: _CounterfactualResult) -> None:
        self._result = result

    def as_dataframe(self) -> pd.DataFrame:
        """Return the counterfactual result as a dataframe"""
        entities = self._result.entities
        features = self._result.getFeatures()

        data = {}
        data["features"] = [f"{entity.as_feature().getName()}" for entity in entities]
        data["proposed"] = [entity.as_feature().value.as_obj() for entity in entities]
        data["original"] = [
            feature.getValue().getUnderlyingObject() for feature in features
        ]
        data["constrained"] = [feature.is_constrained for feature in features]

        dfr = pd.DataFrame.from_dict(data)
        dfr["difference"] = dfr.proposed - dfr.original
        return dfr

    def as_html(self) -> Styler:
        """Returned styled dataframe"""
        return self.as_dataframe().style

    def plot(self) -> None:
        """Plot counterfactual"""
        _df = self.as_dataframe().copy()
        _df = _df[_df["difference"] != 0.0]

        def change_colour(value):
            if value == 0.0:
                colour = ds["neutral_primary_colour"]
            elif value > 0:
                colour = ds["positive_primary_colour"]
            else:
                colour = ds["negative_primary_colour"]
            return colour

        with mpl.rc_context(drcp):
            colour = _df["difference"].transform(change_colour)
            plot = _df[["features", "proposed", "original"]].plot.barh(
                x="features", color={"proposed": colour, "original": "black"}
            )
            plot.set_title("Counterfactual")
            plt.show()


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
        return CounterfactualResult(
            self._explainer.explainAsync(prediction, model).get()
        )


class LimeResults(ExplanationVisualiser):
    """Encapsulate LIME results"""

    def __init__(self, saliencies: Dict[str, Saliency]):
        self._saliencies = saliencies

    def as_dataframe(self) -> pd.DataFrame:
        """Return the LIME result as a dataframe"""
        outputs = self._saliencies.keys()

        data = {}
        for output in outputs:
            pfis = self._saliencies.get(output).getPerFeatureImportance()
            data[f"{output}_features"] = [
                f"{pfi.getFeature().getName()}" for pfi in pfis
            ]
            data[f"{output}_score"] = [pfi.getScore() for pfi in pfis]
            data[f"{output}_value"] = [
                pfi.getFeature().getValue().as_number() for pfi in pfis
            ]
            data[f"{output}_confidence"] = [pfi.getConfidence() for pfi in pfis]

        return pd.DataFrame.from_dict(data)

    def as_html(self) -> Styler:
        """Return styled dataframe"""
        return self.as_dataframe().style

    def map(self):
        """Return saliencies map"""
        return self._saliencies

    def plot(self, decision: str) -> None:
        """Plot saliencies"""
        with mpl.rc_context(drcp):
            dictionary = {}
            for feature_importance in self._saliencies.get(
                decision
            ).getPerFeatureImportance():
                dictionary[
                    feature_importance.getFeature().name
                ] = feature_importance.getScore()

            colours = [
                ds["negative_primary_colour"]
                if i < 0
                else ds["positive_primary_colour"]
                for i in dictionary.values()
            ]
            plt.title(f"LIME explanation of {decision}")
            plt.barh(
                range(len(dictionary)),
                dictionary.values(),
                align="center",
                color=colours,
            )
            plt.yticks(range(len(dictionary)), list(dictionary.keys()))
            plt.tight_layout()
            plt.show()


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

    def explain(self, prediction, model: PredictionProvider) -> LimeResults:
        """Request for a LIME explanation given a prediction and a model"""
        return LimeResults(self._explainer.explainAsync(prediction, model).get())


# pylint: disable=invalid-name
class SHAPResults(ExplanationVisualiser):
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

    def as_dataframe(self) -> pd.DataFrame:
        """Returns SHAP explanation as a dataframe"""

        visualizer_data_frame = pd.DataFrame()
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
            return visualizer_data_frame

    def as_html(self) -> Styler:
        """Print out the SHAP values as a formatted dataframe"""

        def _color_feature_values(feature_values, background_vals):
            """Internal function for the dataframe visualization"""
            formats = []
            for i, feature_value in enumerate(feature_values[1:-1]):
                if feature_value < background_vals[i]:
                    formats.append(f"background-color:{ds['negative_primary_colour']}")
                elif feature_value > background_vals[i]:
                    formats.append(f"background-color:{ds['positive_primary_colour']}")
                else:
                    formats.append(None)
            return [None] + formats + [None]

        visualizer_data_frame = pd.DataFrame()
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
                    name="rwg",
                    colors=[
                        ds["negative_primary_colour"],
                        ds["neutral_primary_colour"],
                        ds["positive_primary_colour"],
                    ],
                ),
                subset=(slice(feature_names[0], feature_names[-1]), "SHAP Value"),
                vmin=-1 * max(np.abs(shap_values)),
                vmax=max(np.abs(shap_values)),
            )
            style.set_caption(f"Explanation of {saliency.getOutput().getName()}")
            return style.apply(
                _color_feature_values,
                background_vals=background_mean_feature_values,
                subset="Feature Value",
                axis=0,
            )

    def candlestick_plot(self) -> None:
        """Plot each SHAP explanation as a candlestick plot"""
        with mpl.rc_context(drcp):
            for i, saliency in enumerate(self.shap_results.getSaliencies()):
                shap_values = [
                    pfi.getScore() for pfi in saliency.getPerFeatureImportance()
                ]
                feature_names = [
                    str(pfi.getFeature().getName())
                    for pfi in saliency.getPerFeatureImportance()
                ]
                fnull = self.shap_results.getFnull().getEntry(i)
                prediction = fnull + sum(shap_values)
                plt.figure()
                pos = fnull
                for j, shap_value in enumerate(shap_values):
                    color = (
                        ds["negative_primary_colour"]
                        if shap_value < 0
                        else ds["positive_primary_colour"]
                    )
                    width = 0.9
                    if j > 0:
                        plt.plot(
                            [j - 0.5, j + width / 2 * 0.99], [pos, pos], color=color
                        )
                    plt.bar(j, height=shap_value, bottom=pos, color=color, width=width)
                    pos += shap_values[j]

                    if j != len(shap_values) - 1:
                        plt.plot(
                            [j - width / 2 * 0.99, j + 0.5], [pos, pos], color=color
                        )

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
