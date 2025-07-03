"""
Wrapper module for TSLIME from aix360.
Original at https://github.com/Trusted-AI/AIX360/
"""

from typing import Callable, List, Union

import pandas as pd
import numpy as np
from aix360.algorithms.tslime import TSLimeExplainer as TSLimeExplainerAIX
from aix360.algorithms.tslime.surrogate import LinearSurrogateModel
from pandas.io.formats.style import Styler
import matplotlib.pyplot as plt

from trustyai.explainers.explanation_results import ExplanationResults
from trustyai.utils.extras.timeseries import TSPerturber


class TSSLIMEResults(ExplanationResults):
    """Wraps TSLimeExplainer results. This object is returned by the :class:`~TSLimeExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, explanation):
        self.explanation = explanation

    def as_dataframe(self) -> pd.DataFrame:
        """Returns the weights as a pandas dataframe."""
        return pd.DataFrame(self.explanation["history_weights"])

    def as_html(self) -> Styler:
        """Returns the explanation as an HTML table."""
        dataframe = self.as_dataframe()
        return dataframe.style

    def plot(self):
        """Plot TSLime explanation for the time-series instance. Based on
        https://github.com/Trusted-AI/AIX360/blob/master/examples/tslime/tslime_univariate_demo.ipynb
        """
        relevant_history = self.explanation["history_weights"].shape[0]
        input_data = self.explanation["input_data"]
        relevant_df = input_data[-relevant_history:]

        plt.figure(layout="constrained")
        plt.plot(relevant_df, label="Input Time Series", marker="o")
        plt.gca().invert_yaxis()

        normalized_weights = (
            self.explanation["history_weights"]
            / np.mean(np.abs(self.explanation["history_weights"]))
        ).flatten()

        plt.bar(
            input_data.index[-relevant_history:],
            normalized_weights,
            0.4,
            label="TSLime Weights (Normalized)",
            color="red",
        )
        plt.axhline(y=0, color="r", linestyle="-", alpha=0.4)
        plt.title("Time Series Lime Explanation Plot")
        plt.legend(bbox_to_anchor=(1.25, 1.0), loc="upper right")
        plt.show()


class TSLimeExplainer(TSLimeExplainerAIX):
    """
    Wrapper for TSLimeExplainer from aix360.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: Callable,
        input_length: int,
        n_perturbations: int = 2000,
        relevant_history: int = None,
        perturbers: List[Union[TSPerturber, dict]] = None,
        local_interpretable_model: LinearSurrogateModel = None,
        random_seed: int = None,
    ):
        super().__init__(
            model=model,
            input_length=input_length,
            n_perturbations=n_perturbations,
            relevant_history=relevant_history,
            perturbers=perturbers,
            local_interpretable_model=local_interpretable_model,
            random_seed=random_seed,
        )

    def explain(self, inputs, outputs=None, **kwargs) -> TSSLIMEResults:
        """
        Explain the model's prediction on X.
        """
        _explanation = super().explain_instance(inputs, y=outputs, **kwargs)
        return TSSLIMEResults(_explanation)
