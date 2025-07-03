"""Explainers.countefactual module"""

# pylint: disable = import-error, too-few-public-methods, wrong-import-order, line-too-long,
# pylint: disable = unused-argument
from typing import Optional, Union, List
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import uuid as _uuid

from trustyai import _default_initializer  # pylint: disable=unused-import
from .explanation_results import ExplanationResults
from trustyai.utils._visualisation import (
    DEFAULT_STYLE as ds,
    DEFAULT_RC_PARAMS as drcp,
)

from trustyai.model import (
    counterfactual_prediction,
    PredictionInput,
    Model,
    GoalCriteria,
)

from trustyai.utils.data_conversions import (
    prediction_object_to_numpy,
    prediction_object_to_pandas,
    OneInputUnionType,
    OneOutputUnionType,
    data_conversion_docstring,
    one_input_convert,
)

from org.kie.trustyai.explainability.local.counterfactual import (
    CounterfactualExplainer as _CounterfactualExplainer,
    CounterfactualResult as _CounterfactualResult,
    SolverConfigBuilder as _SolverConfigBuilder,
    CounterfactualConfig as _CounterfactualConfig,
)
from org.kie.trustyai.explainability.model import (
    DataDistribution,
    PredictionProvider,
)

from org.kie.trustyai.explainability.model.domain import FeatureDomain

from org.optaplanner.core.config.solver.termination import TerminationConfig
from java.lang import Long

SolverConfigBuilder = _SolverConfigBuilder
CounterfactualConfig = _CounterfactualConfig


class CounterfactualResult(ExplanationResults):
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
        return prediction_object_to_numpy(
            [PredictionInput([entity.as_feature() for entity in self._result.entities])]
        )

    @property
    def proposed_features_dataframe(self):
        """Return the proposed feature values found from the counterfactual explanation
        as a Pandas DataFrame.
        """
        return prediction_object_to_pandas(
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

    def plot(self, block=True, call_show=True) -> None:
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
            if call_show:
                plt.show(block=block)


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
    @data_conversion_docstring("one_input", "one_output")
    def explain(
        self,
        inputs: OneInputUnionType,
        model: Union[PredictionProvider, Model],
        goal: Optional[OneOutputUnionType] = None,
        feature_domains: List[FeatureDomain] = None,
        data_distribution: Optional[DataDistribution] = None,
        uuid: Optional[_uuid.UUID] = None,
        timeout: Optional[float] = None,
        criteria: Optional[GoalCriteria] = None,
    ) -> CounterfactualResult:
        r"""Request for a counterfactual explanation given a list of features, goals and a
        :class:`~PredictionProvider`

        Parameters
        ----------
        inputs : {}
            List of input features, as a: {}
        goal : {}
            The desired model outputs to be searched for in the counterfactual explanation.
            These can take the form of a: {}
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI model as generated by :class:`~trustyai.model.Model` or a Java :class:`PredictionProvider`
        feature_domains : List[:class:`FeatureDomain`]
            A list of feature domains (each created by :func:`~trustyai.model.feature_domain()`)
            that define the valid domain of the input features. The ith element of the list defines
            the domain of the ith input feature. If the ith element of this list is ``None``, the
            no domain information will be added to the ith feature. If the ith feature had no
            previously-supplied domain information, it will be taken to be constrained and
            non-variable. If ``feature_domains=None``, no domain information will be added to any
            of the features, thus preserving existing domains if they've been manually added
            previously or holding undomained features constrained.
        data_distribution : Optional[:class:`DataDistribution`]
            The :class:`DataDistribution` to use when sampling the inputs.
        uuid : Optional[:class:`_uuid.UUID`]
            The UUID to use during search.
        timeout : Optional[float]
            The timeout time in seconds of the counterfactual explanation.
        criteria : Optional[:class:`GoalCriteria`]
            An optional custom scoring function, wrapped as a :class:`GoalCriteria`.

        Returns
        -------
        :class:`~CounterfactualResult`
            Object containing the results of the counterfactual explanation.
        """
        feature_names = model.feature_names if isinstance(model, Model) else None
        output_names = model.output_names if isinstance(model, Model) else None
        if goal is None and criteria is None:
            raise ValueError("Either a goal or criteria must be provided.")

        _prediction = counterfactual_prediction(
            input_features=one_input_convert(
                inputs, feature_names=feature_names, feature_domains=feature_domains
            ),
            outputs=goal,
            feature_names=feature_names,
            output_names=output_names,
            data_distribution=data_distribution,
            uuid=uuid,
            timeout=timeout,
            criteria=criteria,
        )

        with Model.NonArrowTransmission(model):
            return CounterfactualResult(
                self._explainer.explainAsync(_prediction, model).get()
            )
