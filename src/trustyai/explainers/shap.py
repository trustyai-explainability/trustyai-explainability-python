"""Explainers.shap module"""

# pylint: disable = import-error, too-few-public-methods, wrong-import-order, line-too-long,
# pylint: disable = unused-argument, consider-using-f-string, invalid-name
from typing import Dict, Optional, Union
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
from jpype import JInt

from trustyai import _default_initializer  # pylint: disable=unused-import
from trustyai.utils._visualisation import DEFAULT_STYLE as ds
from .explanation_results import SaliencyResults
from trustyai.model import simple_prediction, Model
from trustyai.utils.data_conversions import (
    OneInputUnionType,
    OneOutputUnionType,
    ManyInputsUnionType,
    ManyOutputsUnionType,
    many_inputs_convert,
    data_conversion_docstring,
    many_outputs_convert,
)

from org.kie.trustyai.explainability.local.shap import (
    ShapConfig as _ShapConfig,
    ShapKernelExplainer as _ShapKernelExplainer,
)

from org.kie.trustyai.explainability.local.shap.background import (
    RandomGenerator,
    KMeansGenerator,
    CounterfactualGenerator,
)

from org.kie.trustyai.explainability.model import (
    PredictionProvider,
    Saliency,
    PerturbationContext,
)
from java.util import Random


# pylint: disable=invalid-name


class SHAPResults(SaliencyResults):
    """Wraps SHAP results. This object is returned by the :class:`~SHAPExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, saliency_results: SaliencyResults, background):
        """Constructor method. This is called internally, and shouldn't ever need to be used
        manually."""
        self._java_saliency_results = saliency_results
        self.background = background

    def saliency_map(self) -> Dict[str, Saliency]:
        """
        Return a dictionary of found saliencies.

        Returns
        -------
        Dict[str, Saliency]
             A dictionary of :class:`~trustyai.model.Saliency` objects, keyed by output name.
        """
        return {
            entry.getKey(): entry.getValue()
            for entry in self._java_saliency_results.saliencies.entrySet()
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
            for output_name, saliency in self.saliency_map().items()
        }

    def _saliency_to_dataframe(self, saliency, output_name):
        background_mean_feature_values = np.mean(
            [
                [f.getValue().asNumber() for f in pi.getFeatures()]
                for pi in self.background
            ],
            0,
        ).tolist()

        data_rows = []
        for i, pfi in enumerate(saliency.getPerFeatureImportance()[:-1]):
            data_rows.append(
                {
                    "Feature": str(pfi.getFeature().getName().toString()),
                    "Value": pfi.getFeature().getValue().getUnderlyingObject(),
                    "Mean Background Value": background_mean_feature_values[i],
                    "SHAP Value": pfi.getScore(),
                    "Confidence": pfi.getConfidence(),
                }
            )

        fnull = {
            "Feature": "Background",
            "Value": None,
            "Mean Background Value": None,
            "SHAP Value": self.get_fnull()[output_name],
        }

        return pd.DataFrame([fnull] + data_rows)

    def as_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Return the SHAP results as dataframes.

        Returns
        -------
        Dict[str, pandas.DataFrame]
            Dictionary of DataFrames, keyed by output name, containing the results of the SHAP
            explanation. For each model output, the table will contain the following columns:

            * ``Feature``: The name of the feature
            * ``Feature Value``: The value of the feature for this particular input.
            * ``Mean Background Value``: The mean value this feature took in the background
            * ``SHAP Value``: The found SHAP value of this feature.
            * ``Confidence``: The confidence of this explanation as returned by the explainer.

        """
        df_dict = {}
        for output_name, saliency in self.saliency_map().items():
            df_dict[output_name] = self._saliency_to_dataframe(saliency, output_name)
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
            for i, feature_value in enumerate(feature_values[1:]):
                if feature_value < background_vals[i]:
                    formats.append(f"background-color:{ds['negative_primary_colour']}")
                elif feature_value > background_vals[i]:
                    formats.append(f"background-color:{ds['positive_primary_colour']}")
                else:
                    formats.append(None)
            return [None] + formats

        df_dict = {}
        for output_name, saliency in self.saliency_map().items():
            df = self._saliency_to_dataframe(saliency, output_name)
            shap_values = df["SHAP Value"].values[1:]
            background_mean_feature_values = df["Mean Background Value"].values[1:]

            style = df.style.background_gradient(
                LinearSegmentedColormap.from_list(
                    name="rwg",
                    colors=[
                        ds["negative_primary_colour"],
                        ds["neutral_primary_colour"],
                        ds["positive_primary_colour"],
                    ],
                ),
                subset=(slice(1, None), "SHAP Value"),
                vmin=-1 * max(np.abs(shap_values)),
                vmax=max(np.abs(shap_values)),
            )
            style.set_caption(f"Explanation of {output_name}")
            df_dict[output_name] = style.apply(
                _color_feature_values,
                background_vals=background_mean_feature_values,
                subset="Value",
                axis=0,
            )
        return df_dict


class BackgroundGenerator:
    r"""Generate a background for the SHAP explainer

    Generate a background for the SHAP explainer via one of three algorithms:

    * `sample`: Randomly sample a set of provided points
    * `kmeans`: Summarize a set of provided points into k centroids
    * `counterfactual`: Generate a set of background points that meet certain criteria

    """

    @data_conversion_docstring("many_inputs")
    def __init__(self, datapoints: ManyInputsUnionType, feature_domains=None, seed=0):
        r"""Initialize the :class:`BackgroundGenerator`.

        Parameters
        ----------
        datapoints : {}
            The set of datapoints to be used to sample/generate the background, as a: {}
        seed : int
            The random seed to use in the sampling/generation method
        """
        self.datapoints = many_inputs_convert(
            datapoints, feature_domains=feature_domains
        )
        self.feature_domains = feature_domains
        self.seed = 0
        self._jrandom = Random()
        self._jrandom.setSeed(self.seed)

    def sample(self, k=100):
        r"""Randomly sample datapoints.

        Parameters
        ----------
        k : int
            The number of datapoints to select

        Returns
        -------
        :list:`PredictionInput`
            The background dataset to pass to the :class:`~SHAPExplainer`
        """
        perturbation_context = PerturbationContext(self._jrandom, 0)
        return RandomGenerator(self.datapoints, perturbation_context).generate(k)

    def kmeans(self, k=100):
        r"""Use k-means clustering over `datapoints` and return k centroids as the background data
        set.

        Parameters
        ----------
        k : int
            The number of centroids to find

        Returns
        -------
        :list:`PredictionInput`
            The background dataset to pass to the :class:`~SHAPExplainer`
        """
        return KMeansGenerator(self.datapoints, self.seed).generate(k)

    @data_conversion_docstring("many_outputs")
    def counterfactual(
        self,
        goals: ManyOutputsUnionType,
        model: PredictionProvider,
        k_per_goal=100,
        **kwargs,
    ):
        r"""Generate a background via the CounterfactualExplainer. This lets you specify
        exact output values that the background dataset conforms to, and thus set the reference
        point by which all SHAP values compare. For example, if your model is a regression
        model, choosing a counterfactual goal of 0 will create a background dataset where
        :math:'f(x) \approx 0 \forall x \in \text{{background}}`, and as such the SHAP values
        will compare against zero, which is a useful baseline for regression.

        Parameters
        ----------
        goals : {}
            The set of background datapoints as a: {}
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model`
        k_per_goal : int
            The number of background datapoints to generate per goal.
        Keyword Arguments:
            * k_seeds: int
                (default= ``5``) For each goal, a number of starting seeds from `datapoints` are used
                to start the search from. These are the `k_seeds` points within `datapoint`
                whose corresponding outputs are closet to the goal output. Choose a larger
                number to get a more diverse background dataset, but the search might require
                larger `max_attempt_count`, `step_count`, and `timeout_seconds` to get good results.
            * goal_threshold: float
                (default= ``.01``) The distance (percentage) threshold defining whether
                a particular output satisfies the goal. Set to 0 to require an exact match, but
                this will likely require larger `max_attempt_count`, `step_count`,
                and `timeout_seconds` to get good results.
            * chain: boolean
                (default= ``False``) If chaining is set to `true`, found counterfactual datapoints
                will be added to the search seeds for subsequent searches. This is useful when a
                range of counterfactual outputs is desired; for example, if the desired goals are
                [0, 1, 2, 3], whichever goal is closest to the closest point within `datapoints`
                will be searched for first. The found counterfactuals from that search are then
                included in the search for the second-closest goal, and so on. This is especially
                helpful if the extremes of the goal range are far outside the range produced by the
                `datapoints`. If only
            * max_attempt_count: int
                (default= ``5'`) If no valid counterfactual can be found for a starting seed in the
                search, the point is slightly perturbed and search is retried. This parameter sets
                the maximum number of perturbation-retry cycles are allowed during generation.
            * step_count: int
                (default= ``5_000``) The number of datapoints to evaluate during the search
            * timeout_seconds: int
                (default= ``3``) The maximum number of seconds allowed for each counterfactual search.
                This will set the maximum runtime of the search to roughly
                `timeout_seconds` * `max_attempt_count` * `k_per_goal` * `len(goals)`

        Returns
        -------
        :list:`PredictionInput`
            The background dataset to pass to the :class:`~SHAPExplainer`
        """
        if self.feature_domains is None:
            raise AttributeError(
                "Feature domains must be passed to perform"
                " meaningful counterfactual search"
            )
        goals_converted = many_outputs_convert(goals)
        generator = (
            CounterfactualGenerator.builder()
            .withModel(model)
            .withKSeeds(kwargs.get("k_seeds", 5))
            .withRandom(self._jrandom)
            .withTimeoutSeconds(kwargs.get("timeout_seconds", 3))
            .withStepCount(kwargs.get("step_count", 5_000))
            .withGoalThreshold(kwargs.get("goal_threshold", 0.01))
            .withMaxAttemptCount(kwargs.get("max_attempt_count", 5))
            .build()
        )

        if len(goals) == 1:
            background = generator.generate(
                self.datapoints, goals_converted[0], k_per_goal
            )
        else:
            background = generator.generateRange(
                self.datapoints, goals_converted, k_per_goal, kwargs.get("chain", False)
            )
        return background


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

    @data_conversion_docstring("many_inputs")
    def __init__(
        self,
        background: ManyInputsUnionType,
        link_type: Optional[_ShapConfig.LinkType] = None,
        **kwargs,
    ):
        r"""Initialize the :class:`SHAPxplainer`.

        Parameters
        ----------
        background : {}
            The set of background datapoints as a: {}
        link_type : :obj:`~_ShapConfig.LinkType`
            A choice of either ``trustyai.explainers._ShapConfig.LinkType.IDENTITY``
            or ``trustyai.explainers._ShapConfig.LinkType.LOGIT``. If the model output is a
            probability, choosing the ``LOGIT`` link will rescale explanations into log-odds units.
            Otherwise, choose ``IDENTITY``.
        Keyword Arguments:
            * samples: int
                (default= ``None``) The number of samples to use when computing SHAP values. Higher
                values will increase explanation accuracy, at the  cost of runtime. If none,
                samples will equal 2048 + 2*n_features
            * seed: int
                (default= ``0``) The random seed to be used when generating explanations.
            * batchSize: int
                (default= ``20``) The number of batches passed to the PredictionProvider at once.
                When using :class:`~Model` with `disable_arrow=True` this parameter has no effect.
                If `arrow=True`, `batch_sizes` of around
                :math:`\frac{{2000}}{{\mathtt{{len(background)}}}}` can produce significant
                performance gains.
            * trackCounterfactuals : bool
                (default= ``False``) Keep track of produced byproduct counterfactuals during SHAP run.

        Returns
        -------
        :class:`~SHAPResults`
            Object containing the results of the SHAP explanation.
        """
        if not link_type:
            link_type = _ShapConfig.LinkType.IDENTITY
        self._jrandom = Random()
        self._jrandom.setSeed(kwargs.get("seed", 0))
        self._raw_background = background
        perturbation_context = PerturbationContext(self._jrandom, 0)

        self._configbuilder = (
            _ShapConfig.builder()
            .withLink(link_type)
            .withBatchSize(kwargs.get("batch_size", 20))
            .withPC(perturbation_context)
            .withTrackCounterfactuals(kwargs.get("track_counterfactuals", False))
        )
        if kwargs.get("samples") is not None:
            self._configbuilder.withNSamples(JInt(kwargs["samples"]))

    @data_conversion_docstring("one_input", "one_output")
    def explain(
        self,
        inputs: OneInputUnionType,
        outputs: OneOutputUnionType,
        model: Union[PredictionProvider, Model],
    ) -> SHAPResults:
        """Produce a SHAP explanation.

        Parameters
        ----------
        inputs : {}
            The input features to the model, as a: {}
        outputs : {}
            The corresponding model outputs for the provided features, that is,
            ``outputs = model(input_features)``. These can take the form of a: {}
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model`

        Returns
        -------
        :class:`~SHAPResults`
            Object containing the results of the SHAP explanation.
        """

        feature_names = model.feature_names if isinstance(model, Model) else None
        output_names = model.output_names if isinstance(model, Model) else None
        _prediction = simple_prediction(inputs, outputs, feature_names, output_names)
        _background = many_inputs_convert(self._raw_background, feature_names)
        config = self._configbuilder.withBackground(_background).build()
        explainer = _ShapKernelExplainer(config)
        with Model.ArrowTransmission(model, inputs):
            return SHAPResults(
                explainer.explainAsync(_prediction, model).get(), _background
            )
