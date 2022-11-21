"""Explainers module"""
# pylint: disable = import-error, too-few-public-methods, wrong-import-order, line-too-long,
# pylint: disable = unused-argument
from typing import Dict, Optional, List, Union
import matplotlib.pyplot as plt
import matplotlib as mpl
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
from jpype import JInt
import uuid as _uuid

from trustyai import _default_initializer  # pylint: disable=unused-import
from trustyai.utils._visualisation import (
    ExplanationVisualiser,
    DEFAULT_STYLE as ds,
    DEFAULT_RC_PARAMS as drcp,
    bold_red_html,
    bold_green_html,
    output_html,
    feature_html,
)

from trustyai.model import (
    counterfactual_prediction,
    feature,
    Dataset,
    PredictionInput,
    simple_prediction,
)

from org.kie.trustyai.explainability.local.counterfactual import (
    CounterfactualExplainer as _CounterfactualExplainer,
    CounterfactualResult as _CounterfactualResult,
    SolverConfigBuilder as _SolverConfigBuilder,
    CounterfactualConfig as _CounterfactualConfig,
)
from org.kie.trustyai.explainability.local.lime import (
    LimeConfig as _LimeConfig,
    LimeExplainer as _LimeExplainer,
)
from org.kie.trustyai.explainability.local.shap import (
    ShapConfig as _ShapConfig,
    ShapKernelExplainer as _ShapKernelExplainer,
)
from org.kie.trustyai.explainability.model import (
    DataDistribution,
    EncodingParams,
    Feature,
    Output,
    PredictionOutput,
    PredictionProvider,
    Saliency,
    SaliencyResults,
    PerturbationContext,
)
from org.optaplanner.core.config.solver.termination import TerminationConfig
from java.lang import Long
from java.util import Random

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

    @property
    def proposed_features_array(self):
        """Return the proposed feature values found from the counterfactual explanation
        as a Numpy array.
        """
        return Dataset.prediction_object_to_numpy(
            [PredictionInput([entity.as_feature() for entity in self._result.entities])]
        )

    @property
    def proposed_features_dataframe(self):
        """Return the proposed feature values found from the counterfactual explanation
        as a Pandas DataFrame.
        """
        return Dataset.prediction_object_to_pandas(
            [PredictionInput([entity.as_feature() for entity in self._result.entities])]
        )

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

    # pylint: disable=too-many-arguments
    def explain(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, List[Feature], PredictionInput],
        goal: Union[np.ndarray, pd.DataFrame, List[Output], PredictionOutput],
        model: PredictionProvider,
        data_distribution: Optional[DataDistribution] = None,
        uuid: Optional[_uuid.UUID] = None,
        timeout: Optional[float] = None,
    ) -> CounterfactualResult:
        """Request for a counterfactual explanation given a list of features, goals and a
        :class:`~PredictionProvider`

        Parameters
        ----------
        inputs : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Feature`], or :class:`PredictionInput`
            List of input features, as a:

            * Numpy array of shape ``[1, n_features]``
            * Pandas DataFrame with 1 row and ``n_features`` columns
            * A List of TrustyAI :class:`Feature`, as created by the :func:`~feature` function
            * A TrustyAI :class:`PredictionInput`

        goal : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Output`], or :class:`PredictionOutput`
            The desired model outputs to be searched for in the counterfactual explanation.
            These can take the form of a:

            * Numpy array of shape ``[1, n_outputs]``
            * Pandas DataFrame with 1 row and ``n_outputs`` columns
            * A List of TrustyAI :class:`Output`, as created by the :func:`~output` function
            * A TrustyAI :class:`PredictionOutput`

        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model` or
             :class:`~trustyai.model.ArrowModel`.

        data_distribution : Optional[:class:`DataDistribution`]
            The :class:`DataDistribution` to use when sampling the inputs.
        uuid : Optional[:class:`_uuid.UUID`]
            The UUID to use during search.
        timeout : Optional[float]
                The timeout time in seconds of the counterfactual explanation.
        Returns
        -------
        :class:`~CounterfactualResult`
            Object containing the results of the counterfactual explanation.
        """
        _prediction = counterfactual_prediction(
            input_features=inputs,
            outputs=goal,
            data_distribution=data_distribution,
            uuid=uuid,
            timeout=timeout,
        )
        return CounterfactualResult(
            self._explainer.explainAsync(_prediction, model).get()
        )


class LimeResults(ExplanationVisualiser):
    """Wraps LIME results. This object is returned by the :class:`~LimeExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, saliencyResults: SaliencyResults):
        """Constructor method. This is called internally, and shouldn't ever need to be used
        manually."""
        self._saliency_results = saliencyResults

    def map(self) -> Dict[str, Saliency]:
        """
        Return a dictionary of found saliencies.

        Returns
        -------
        Dict[str, Saliency]
             A dictionary of :class:`~trustyai.model.Saliency` objects, keyed by output name.
        """
        return {
            entry.getKey(): entry.getValue()
            for entry in self._saliency_results.saliencies.entrySet()
        }

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
        outputs = self.map().keys()

        data = {}
        for output in outputs:
            pfis = self.map().get(output).getPerFeatureImportance()
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

    def plot(self, decision: str) -> None:
        """Plot the LIME saliencies."""
        with mpl.rc_context(drcp):
            dictionary = {}
            for feature_importance in (
                self.map().get(decision).getPerFeatureImportance()
            ):
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

    def _get_bokeh_plot_dict(self):
        plot_dict = {}
        for output_name, value in self.map().items():
            lime_data_source = pd.DataFrame(
                [
                    {
                        "feature": str(pfi.getFeature().getName()),
                        "saliency": pfi.getScore(),
                    }
                    for pfi in value.getPerFeatureImportance()
                ]
            )
            lime_data_source["color"] = lime_data_source["saliency"].apply(
                lambda x: ds["positive_primary_colour"]
                if x >= 0
                else ds["negative_primary_colour"]
            )
            lime_data_source["saliency_colored"] = lime_data_source["saliency"].apply(
                lambda x: (bold_green_html if x >= 0 else bold_red_html)(
                    "{:.2f}".format(x)
                )
            )

            lime_data_source["color_faded"] = lime_data_source["saliency"].apply(
                lambda x: ds["positive_primary_colour_faded"]
                if x >= 0
                else ds["negative_primary_colour_faded"]
            )
            source = ColumnDataSource(lime_data_source)
            htool = HoverTool(
                names=["bars"],
                tooltips="<h3>LIME</h3> {} saliency to {}: @saliency_colored".format(
                    feature_html("@feature"), output_html(output_name)
                ),
            )
            bokeh_plot = figure(
                sizing_mode="stretch_both",
                title="Lime Feature Importances",
                y_range=lime_data_source["feature"],
                tools=[htool],
            )
            bokeh_plot.hbar(
                y="feature",
                left=0,
                right="saliency",
                fill_color="color_faded",
                line_color="color",
                hover_color="color",
                color="color",
                height=0.75,
                name="bars",
                source=source,
            )
            bokeh_plot.line([0, 0], [0, len(lime_data_source)], color="#000")
            bokeh_plot.xaxis.axis_label = "Saliency Value"
            bokeh_plot.yaxis.axis_label = "Feature"
            plot_dict[output_name] = bokeh_plot
        return plot_dict


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
        track_counterfactuals=False,
        normalise_weights=False,
        use_wlr_model=True,
        **kwargs
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
            .withUseWLRLinearModel(use_wlr_model)
            .withTrackCounterfactuals(track_counterfactuals)
        )

        self._explainer = _LimeExplainer(self._lime_config)

    def explain(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, List[Feature], PredictionInput],
        outputs: Union[np.ndarray, pd.DataFrame, List[Output], PredictionOutput],
        model: PredictionProvider,
    ) -> LimeResults:
        """Produce a LIME explanation.

        Parameters
        ----------
        inputs : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Feature`], or :class:`PredictionInput`
            The input features to the model, as a:

            * Numpy array of shape ``[1, n_features]``
            * Pandas DataFrame with 1 row and ``n_features`` columns
            * A List of TrustyAI :class:`Feature`, as created by the :func:`~feature` function
            * A TrustyAI :class:`PredictionInput`

        outputs : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Output`], or :class:`PredictionOutput`
            The corresponding model outputs for the provided features, that is,
            ``outputs = model(input_features)``. These can take the form of a:

            * Numpy array of shape ``[1, n_outputs]``
            * Pandas DataFrame with 1 row and ``n_outputs`` columns
            * A List of TrustyAI :class:`Output`, as created by the :func:`~output` function
            * A TrustyAI :class:`PredictionOutput`

        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model`
            or :class:`~trustyai.model.ArrowModel`.

        Returns
        -------
        :class:`~LimeResults`
            Object containing the results of the LIME explanation.
        """
        _prediction = simple_prediction(inputs, outputs)
        return LimeResults(self._explainer.explainAsync(_prediction, model).get())


# pylint: disable=invalid-name
class SHAPResults(ExplanationVisualiser):
    """Wraps SHAP results. This object is returned by the :class:`~SHAPExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, saliency_results: SaliencyResults, background):
        """Constructor method. This is called internally, and shouldn't ever need to be used
        manually."""
        self._saliency_results = saliency_results
        self.background = background

    def get_saliencies(self) -> Dict[str, Saliency]:
        """
        Return a dictionary of found saliencies.

        Returns
        -------
        Dict[str, Saliency]
             A dictionary of :class:`~trustyai.model.Saliency` objects, keyed by output name.
        """
        return {
            entry.getKey(): entry.getValue()
            for entry in self._saliency_results.saliencies.entrySet()
        }

    def get_fnull(self):
        """
        Return the list of the found fnulls (y-intercepts) of the SHAP explanations

        Returns
        -------
        Array[float]
             An array of the y-intercepts, in order of the model outputs.
        """
        return {
            output_name: saliency.getPerFeatureImportance()[-1].getScore()
            for output_name, saliency in self.get_saliencies().items()
        }

    def _saliency_to_dataframe(self, saliency, output_name):
        background_mean_feature_values = np.mean(
            [
                [f.getValue().asNumber() for f in pi.getFeatures()]
                for pi in self.background
            ],
            0,
        ).tolist()
        feature_values = [
            pfi.getFeature().getValue().asNumber()
            for pfi in saliency.getPerFeatureImportance()[:-1]
        ]
        shap_values = [
            pfi.getScore() for pfi in saliency.getPerFeatureImportance()[:-1]
        ]
        feature_names = [
            str(pfi.getFeature().getName())
            for pfi in saliency.getPerFeatureImportance()[:-1]
        ]

        columns = ["Mean Background Value", "Feature Value", "SHAP Value"]
        visualizer_data_frame = pd.DataFrame(
            [background_mean_feature_values, feature_values, shap_values],
            index=columns,
            columns=feature_names,
        ).T
        fnull = self.get_fnull()[output_name]

        return (
            pd.concat(
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
            ),
            feature_names,
            shap_values,
            background_mean_feature_values,
        )

    def as_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Return the SHAP results as dataframes.

        Returns
        -------
        Dict[str, pandas.DataFrame]
            Dictionary of DataFrames, keyed by output name, containing the results of the SHAP
            explanation. For each model output, the table will contain the following columns,
            indexed by feature name:

            * ``Mean Background Value``: The mean value this feature took in the background
            * ``Feature Value``: The value of the feature for this particular input.
            * ``SHAP Value``: The found SHAP value of this feature.
        """
        df_dict = {}
        for output_name, saliency in self.get_saliencies().items():
            df_dict[output_name] = self._saliency_to_dataframe(saliency, output_name)[0]
        return df_dict

    def as_html(self) -> Dict[str, pd.io.formats.style.Styler]:
        """
        Return the SHAP results as Pandas Styler objects.

        Returns
        -------
        Dict[str, pandas.Styler]
            Dictionary of stylers keyed by output name. Each styler containing the results of the
            SHAP explanation for that particular output, in the same
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

        df_dict = {}
        for i, (output_name, saliency) in enumerate(self.get_saliencies().items()):
            (
                df,
                feature_names,
                shap_values,
                background_mean_feature_values,
            ) = self._saliency_to_dataframe(saliency, i)
            style = df.style.background_gradient(
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
            style.set_caption(f"Explanation of {output_name}")
            df_dict[output_name] = style.apply(
                _color_feature_values,
                background_vals=background_mean_feature_values,
                subset="Feature Value",
                axis=0,
            )
        return df_dict

    def candlestick_plot(self) -> None:
        """Visualize the SHAP explanation of each output as a set of candlestick plots,
        one per output."""
        with mpl.rc_context(drcp):
            for output_name, saliency in self.get_saliencies().items():
                shap_values = [
                    pfi.getScore() for pfi in saliency.getPerFeatureImportance()[:-1]
                ]
                feature_names = [
                    str(pfi.getFeature().getName())
                    for pfi in saliency.getPerFeatureImportance()[:-1]
                ]
                fnull = self.get_fnull()[output_name]
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
                plt.title(f"Explanation of {output_name}")
                plt.show()

    def _get_bokeh_plot_dict(self):
        plots = {}
        for output_name, value in self.get_saliencies().items():
            fnull = self.get_fnull()[output_name]

            # create dataframe of plot values
            data_source = pd.DataFrame(
                [
                    {
                        "feature": str(pfi.getFeature().getName()),
                        "saliency": pfi.getScore(),
                    }
                    for pfi in value.getPerFeatureImportance()[:-1]
                ]
            )
            prediction = fnull + data_source["saliency"].sum()

            data_source["color"] = data_source["saliency"].apply(
                lambda x: ds["positive_primary_colour"]
                if x >= 0
                else ds["negative_primary_colour"]
            )
            data_source["color_faded"] = data_source["saliency"].apply(
                lambda x: ds["positive_primary_colour_faded"]
                if x >= 0
                else ds["negative_primary_colour_faded"]
            )
            data_source["index"] = data_source.index
            data_source["saliency_text"] = data_source["saliency"].apply(
                lambda x: (bold_red_html if x <= 0 else bold_green_html)(
                    "{:.2f}".format(x)
                )
            )
            data_source["bottom"] = pd.Series(
                [fnull] + data_source["saliency"].iloc[0:-1].tolist()
            ).cumsum()
            data_source["top"] = data_source["bottom"] + data_source["saliency"]

            # create hovertools
            htool_fnull = HoverTool(
                names=["fnull"],
                tooltips=("<h3>SHAP</h3>Baseline {}: {}").format(
                    output_name, output_html("{:.2f}".format(fnull))
                ),
                line_policy="interp",
            )
            htool_pred = HoverTool(
                names=["pred"],
                tooltips=("<h3>SHAP</h3>Predicted {}: {}").format(
                    output_name, output_html("{:.2f}".format(prediction))
                ),
                line_policy="interp",
            )
            htool_bars = HoverTool(
                names=["bars"],
                tooltips="<h3>SHAP</h3> {} contributions to {}: @saliency_text".format(
                    feature_html("@feature"), output_html(output_name)
                ),
            )

            # create plot
            bokeh_plot = figure(
                sizing_mode="stretch_both",
                title="SHAP Feature Contributions",
                x_range=data_source["feature"],
                tools=[htool_pred, htool_fnull, htool_bars],
            )

            # add fnull and background lines
            line_data_source = ColumnDataSource(
                pd.DataFrame(
                    [
                        {"x": 0, "pred": prediction},
                        {"x": len(data_source), "pred": prediction},
                    ]
                )
            )
            fnull_data_source = ColumnDataSource(
                pd.DataFrame(
                    [{"x": 0, "fnull": fnull}, {"x": len(data_source), "fnull": fnull}]
                )
            )

            bokeh_plot.line(
                x="x",
                y="fnull",
                line_color="#999",
                hover_line_color="#333",
                line_width=2,
                hover_line_width=4,
                line_dash="dashed",
                name="fnull",
                source=fnull_data_source,
            )
            bokeh_plot.line(
                x="x",
                y="pred",
                line_color="#999",
                hover_line_color="#333",
                line_width=2,
                hover_line_width=4,
                name="pred",
                source=line_data_source,
            )

            # create candlestick plot lines
            bokeh_plot.line(
                x=[0.5, 1],
                y=data_source.iloc[0]["top"],
                color=data_source.iloc[0]["color"],
            )
            for i in range(1, len(data_source)):
                # bar left line
                bokeh_plot.line(
                    x=[i, i + 0.5],
                    y=data_source.iloc[i]["bottom"],
                    color=data_source.iloc[i]["color"],
                )
                # bar right line
                if i != len(data_source) - 1:
                    bokeh_plot.line(
                        x=[i + 0.5, i + 1],
                        y=data_source.iloc[i]["top"],
                        color=data_source.iloc[i]["color"],
                    )

            # create candles
            bokeh_plot.vbar(
                x="feature",
                bottom="bottom",
                top="top",
                hover_color="color",
                color="color_faded",
                width=0.75,
                name="bars",
                source=data_source,
            )
            bokeh_plot.yaxis.axis_label = str(output_name)
            plots[output_name] = bokeh_plot
        return plots


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
        background: Union[np.ndarray, pd.DataFrame, List[PredictionInput]],
        samples=None,
        batch_size=20,
        seed=0,
        link_type: Optional[_ShapConfig.LinkType] = None,
        track_counterfactuals=False,
        **kwargs,
    ):
        r"""Initialize the :class:`SHAPxplainer`.

        Parameters
        ----------
        background : :class:`numpy.array`, :class:`Pandas.DataFrame`
        or List[:class:`PredictionInput]
            The set of background datapoints as an array, dataframe of shape
            ``[n_datapoints, n_features]``, or list of TrustyAI PredictionInputs.
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
        link_type : :obj:`~_ShapConfig.LinkType`
            A choice of either ``trustyai.explainers._ShapConfig.LinkType.IDENTITY``
            or ``trustyai.explainers._ShapConfig.LinkType.LOGIT``. If the model output is a
            probability, choosing the ``LOGIT`` link will rescale explanations into log-odds units.
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
        perturbation_context = PerturbationContext(self._jrandom, 0)

        if isinstance(background, np.ndarray):
            self.background = Dataset.numpy_to_prediction_object(background, feature)
        elif isinstance(background, pd.DataFrame):
            self.background = Dataset.df_to_prediction_object(background, feature)
        elif isinstance(background[0], PredictionInput):
            self.background = background
        else:
            raise AttributeError(
                "Unsupported background type: {}".format(type(background))
            )

        self._configbuilder = (
            _ShapConfig.builder()
            .withLink(link_type)
            .withBatchSize(batch_size)
            .withPC(perturbation_context)
            .withBackground(self.background)
            .withTrackCounterfactuals(track_counterfactuals)
        )
        if samples is not None:
            self._configbuilder.withNSamples(JInt(samples))
        self._config = self._configbuilder.build()
        self._explainer = _ShapKernelExplainer(self._config)

    def explain(
        self,
        inputs: Union[np.ndarray, pd.DataFrame, List[Feature], PredictionInput],
        outputs: Union[np.ndarray, pd.DataFrame, List[Output], PredictionOutput],
        model: PredictionProvider,
    ) -> SHAPResults:
        """Produce a SHAP explanation.

        Parameters
        ----------
        inputs : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Feature`], or :class:`PredictionInput`
            The input features to the model, as a:

            * Numpy array of shape ``[1, n_features]``
            * Pandas DataFrame with 1 row and ``n_features`` columns
            * A List of TrustyAI :class:`Feature`, as created by the :func:`~feature` function
            * A TrustyAI :class:`PredictionInput`

        outputs : :class:`numpy.ndarray`, :class:`pandas.DataFrame`, List[:class:`Output`], or :class:`PredictionOutput`
            The corresponding model outputs for the provided features, that is,
            ``outputs = model(input_features)``. These can take the form of a:

            * Numpy array of shape ``[1, n_outputs]``
            * Pandas DataFrame with 1 row and ``n_outputs`` columns
            * A List of TrustyAI :class:`Output`, as created by the :func:`~output` function
            * A TrustyAI :class:`PredictionOutput`
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model` or
            :class:`~trustyai.model.ArrowModel`.

        Returns
        -------
        :class:`~SHAPResults`
            Object containing the results of the SHAP explanation.
        """
        _prediction = simple_prediction(inputs, outputs)
        return SHAPResults(
            self._explainer.explainAsync(_prediction, model).get(), self.background
        )
