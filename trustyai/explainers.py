"""Explainers module"""
# pylint: disable = import-error, too-few-public-methods
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
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
    """Wraps Counterfactual results. This object is returned by the
    :class:`~CounterfactualExplainer`, and provides a variety of methods to visualize and interact
    with the results of the counterfactual explanation.
    """

    def __init__(self, result: _CounterfactualResult) -> None:
        """Constructor method. This is called internally, and shouldn't ever need to be
        used manually."""
        self._result = result

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return the counterfactual result as a dataframe

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the results of the counterfactual explanation, containing the
            following columns:

            * ``Features``: The names of each input feature.
            * ``Proposed``: The found values of the features.
            * ``Original``: The original feature values.
            * ``Constrained``: Whether this feature was constrained (held fixed) during the search.
            * ``Difference``: The difference between the proposed and original values.
        """
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

    def as_html(self) -> pd.io.formats.style.Styler:
        """
        Return the counterfactual result as a Pandas Styler object.

        Returns
        -------
        pandas.Styler
            Styler containing the results of the counterfactual explanation, in the same
            schema as in :func:`as_dataframe`. Currently, no default styles are applied
            in this particular function, making it equivalent to :code:`self.as_dataframe().style`.
        """
        return self.as_dataframe().style

    def plot(self) -> None:
        """
        Plot the counterfactual result.
        """
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
    """*"How do I get the result I want?"*

    The CounterfactualExplainer class seeks to answer this question by exploring "what-if"
    scenarios. Given some initial input and desired outcome, the counterfactual explainer tries to
    find a set of nearby inputs that produces the desired outcome. Mathematically, if we have a
    model :math:`f`, some input :math:`x` and a desired model output :math:`y'`, the counterfactual
    explainer finds some nearby input :math:`x'` such that :math:`f(x') = y'`.
    """

    def __init__(self, steps=10_000):
        """
        Build a new counterfactual explainer.

        Parameters
        ----------
        steps: int
            The number of search steps to perform during the counterfactual search.
        """
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
        """Request for a counterfactual explanation given a :class:`~CounterfactualPrediction` and a
        :class:`~PredictionProvider`

        Parameters
        ----------
        prediction : :obj:`trustyai.model.CounterfactualPrediction`
            The counterfactual prediction as returned by
            :func:`~trustyai.model.counterfactual_prediction`. This object wraps both the initial
            input and desired output together.
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model` or
             :class:`~trustyai.model.ArrowModel`.

        Returns
        -------
        :class:`~CounterfactualResult`
            Object containing the results of the counterfactual explanation.
        """
        return CounterfactualResult(
            self._explainer.explainAsync(prediction, model).get()
        )


class LimeResults(ExplanationVisualiser):
    """Wraps LIME results. This object is returned by the :class:`~LimeExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, saliencies: Dict[str, Saliency]):
        """Constructor method. This is called internally, and shouldn't ever need to be used
        manually."""
        self._saliencies = saliencies

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return the LIME result as a dataframe.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the results of the LIME explanation. For each model output, the
            table will contain the following columns:

            * ``${output_name}_features``: The names of each input feature.
            * ``${output_name}_score``: The LIME saliency of this feature.
            * ``${output_name}_value``: The original value of each feature.
            * ``${output_name}_confidence``: The confidence of the reported saliency.
        """
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

    def as_html(self) -> pd.io.formats.style.Styler:
        """
        Return the LIME result as a Pandas Styler object.

        Returns
        -------
        pandas.Styler
            Styler containing the results of the LIME explanation, in the same
            schema as in :func:`as_dataframe`. Currently, no default styles are applied
            in this particular function, making it equivalent to :code:`self.as_dataframe().style`.
        """
        return self.as_dataframe().style

    def map(self) -> Dict[str, Saliency]:
        """
        Return the dictionary of the found saliencies.

        Returns
        -------
        Dict[str, Saliency]
             A dictionary keyed by output name, and the values will be the corresponding
              :class:`~trustyai.model.Saliency` object.
        """
        return self._saliencies

    def plot(self, decision: str) -> None:
        """Plot the LIME saliencies."""
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
    """*"Which features were most important to the results?"*

    LIME (`Local Interpretable Model-agnostic Explanations <https://arxiv.org/abs/1602.04938>`_)
    seeks to answer this question via providing *saliencies*, weights associated with each input
    feature that describe how strongly said feature contributed to the model's output.
    """

    def __init__(
        self,
        perturbations=1,
        seed=0,
        samples=10,
        penalise_sparse_balance=True,
        normalise_weights=True,
    ):
        """Initialize the :class:`LimeExplainer`.

        Parameters
        ----------
        perturbations: int
            The starting number of feature perturbations within the explanation process.
        seed: int
            The random seed to be used.
        samples: int
            Number of samples to be generated for the local linear model training.
        penalise_sparse_balance : bool
            Whether to penalise features that are likely to produce linearly inseparable outputs.
            This can improve the efficacy and interpretability of the outputted saliencies.
        normalise_weights : bool
            Whether to normalise the saliencies generated by LIME. If selected, saliencies will be
            normalized between 0 and 1.
        """
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
        """Produce a LIME explanation.

        Parameters
        ----------
        prediction : :obj:`~trustyai.model.Prediction`
            A :class:`~trustyai.model.Prediction` as returned
            by :func:`~trustyai.model.simple_prediction`. This object wraps the
            original model input and output together.
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model`
            or :class:`~trustyai.model.ArrowModel`.

        Returns
        -------
        :class:`~LimeResults`
            Object containing the results of the LIME explanation.
        """
        return LimeResults(self._explainer.explainAsync(prediction, model).get())


# pylint: disable=invalid-name
class SHAPResults(ExplanationVisualiser):
    """Wraps SHAP results. This object is returned by the :class:`~SHAPExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, shap_results, background):
        """Constructor method. This is called internally, and shouldn't ever need to be used
        manually."""
        self.shap_results = shap_results
        self.background = background

    def getSaliencies(self) -> List[Saliency]:
        """
        Return the list of the found saliencies.

        Returns
        -------
        List[Saliency]
             A list of :class:`~trustyai.model.Saliency` objects, in the order of the model outputs.
        """
        return self.shap_results.getSaliencies()

    def getFnull(self):
        """
        Return the list of the found fnulls (y-intercepts) of the SHAP explanations

        Returns
        -------
        Array[float]
             An array of the y-intercepts, in order of the model outputs.
        """
        return self.shap_results.getFnull()

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return the SHAP result as a dataframe.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the results of the SHAP explanation. For each model output,
            the table will contain the following columns, indexed by feature name:

            * ``Mean Background Value``: The mean value this feature took in the background
            * ``Feature Value``: The value of the feature for this particular input.
            * ``SHAP Value``: The found SHAP value of this feature.
        """

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

    def as_html(self) -> pd.io.formats.style.Styler:
        """
        Return the SHAP result as a Pandas Styler object.

        Returns
        -------
        pandas.Styler
            Styler containing the results of the SHAP explanation, in the same
            schema as in :func:`as_dataframe`. This will:

            * Color each ``Feature Value`` based on how it compares to the corresponding
              ``Mean Background Value``.
            * Color each ``SHAP Value`` based on how their magnitude.
        """

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
        """Visualize the SHAP explanation of each output as a set of candlestick plots,
        one per output."""
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
    r"""*"By how much did each feature contribute to the outputs?"*

    SHAP (`SHapley Additive exPlanations <https://arxiv.org/abs/1705.07874>`_) seeks to answer
    this question via providing SHAP values that provide an additive explanation of the model
    output; essentially a `receipt` for the model's output. SHAP does this by finding an
    *additive explanatory model* :math:`g` of the form:

    .. math::
        f(x) = \phi_0 + \phi_1 x'_1 + \phi_2 x'_2 + \dots + \phi_n x'_n

    where :math:`x'_1, \dots, x'_n` are binary values that indicate whether the :math:`n` th
    feature ispresent or absent and :math:`\phi_1, \dots, \phi_n` are those features' corresponding
    SHAP values. :math:`\phi_0` is the *fnull* of the model, indicating the model's latent
    output in the absence of all features; functionally, the y-intercept of the explanatory model.

    What all this means is that a feature's exact contribution to the output can be seen as its
    SHAP value, and the original model output can be recovered by summing up the fnull with all
    SHAP values.

    To operate, SHAP also needs access to a *background dataset*, a set of representative input
    datapoints that captures the model's "normal behavior". All SHAP values are implicitly
    comparisons against to the background data, i.e., By how much did each feature contribute to
    the outputs, as compared to the background inputs?*
    """

    def __init__(
        self,
        background: List[_PredictionInput],
        samples=None,
        batch_size=20,
        seed=0,
        perturbations=0,
        link_type: Optional[_ShapConfig.LinkType] = None,
    ):
        r"""Initialize the :class:`SHAPxplainer`.

        Parameters
        ----------
        background : list[:obj:`~trustyai.model.PredictionInput`]
            The set of background datapoints
        samples: int
            The number of samples to use when computing SHAP values. Higher values will increase
            explanation accuracy, at the  cost of runtime.
        batch_size: int
            The number of batches passed to the PredictionProvider at once. When using a
            :class:`~Model` in the :func:`explain` function, this parameter has no effect. With an
            :class:`~ArrowModel`, `batch_sizes` of around
            :math:`\frac{2000}{\mathtt{len(background)}}` can produce significant
            performance gains.
        seed: int
            The random seed to be used when generating explanations.
        perturbations: int
            This argument has no effect and will be removed shortly, ignore.
        link_type : :obj:`~_ShapConfig.LinkType`
            A choice of either ``trustyai.explainers._ShapConfig.LinkType.IDENTITY``
            or ``trustyai.explainers._ShapConfig.LinkType.LOGIT``. If the model output is a
            probability, choosing the ``LOGIT`` link will rescale explanations into log-oods units.
            Otherwise, choose ``IDENTITY``.

        Returns
        -------
        :class:`~SHAPResults`
            Object containing the results of the SHAP explanation.
        """
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
        """Produce a SHAP explanation.

        Parameters
        ----------
        prediction : :obj:`~trustyai.model.Prediction`
            A :class:`~trustyai.model.Prediction` as returned
            by :func:`~trustyai.model.simple_prediction`. This object wraps the original model input
            and output together.
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model` or
            :class:`~trustyai.model.ArrowModel`.

        Returns
        -------
        :class:`~SHAPResults`
            Object containing the results of the SHAP explanation.
        """
        return SHAPResults(
            self._explainer.explainAsync(prediction, model).get(), self.background
        )
