"""Explainers module"""
# pylint: disable=duplicate-code
from jpype import (
    JImplements,
    JOverride,
)

from .counterfactuals import CounterfactualResult, CounterfactualExplainer
from .lime import LimeExplainer, LimeResults
from .shap import SHAPExplainer, SHAPResults, BackgroundGenerator


@JImplements("org.kie.trustyai.explainability.local.LocalExplainer", deferred=True)
class LocalExplainer:
    """
    Wraps java LocalExplainer interface, delegating the generation of explanations
    to either LimeExplainer or SHAPExplainer.
    """

    def __init__(self, wrapped):
        """
        Parameters
        ----------
        wrapped: Union[trustyai.explainer.LimeExplainer, trustyai.explainer.SHAPExplainer]
            wrapped explainer
        """
        self.wrapped = wrapped

    @JOverride
    def explainAsync(self, prediction, model):
        """
        Parameters
        ----------
        prediction: trustyai.model.SimplePrediction
            the prediction to explain
        model: trustyai.model.PredictionProvider
            the model used to generate the prediction

        LocalExplainer explainAsync method implementation is delegated to the
        underlying explainer (LIME or SHAP).
        """
        return self.wrapped.explainAsync(prediction, model)
