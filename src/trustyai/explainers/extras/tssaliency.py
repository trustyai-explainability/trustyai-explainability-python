from typing import Callable, List

import pandas as pd
import numpy as np
from aix360.algorithms.tssaliency import TSSaliencyExplainer as TSSaliencyExplainerAIX
from pandas.io.formats.style import Styler
import matplotlib.pyplot as plt

from trustyai.explainers.explanation_results import ExplanationResults


class TSSaliencyResults(ExplanationResults):
    """Wraps TSSaliency results. This object is returned by the :class:`~TSSaliencyExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, explanation):
        self.explanation = explanation

    def as_dataframe(self) -> pd.DataFrame:
        saliencies = self.explanation['saliency'].reshape(-1)
        df = pd.DataFrame(saliencies, columns=self.explanation['feature_names'])
        return df

    def as_html(self) -> Styler:
        """Returns the explanation as an HTML table."""
        dataframe = self.as_dataframe()
        return dataframe.style

    def plot(self):
        """Plot tssaliency explanation for the test point
        Based on https://github.com/Trusted-AI/AIX360/blob/master/examples/tssaliency/tssaliency_univariate_demo.ipynb"""
        max_abs = np.max(np.abs(self.explanation['saliency']))

        plt.imshow(self.explanation['saliency'][np.newaxis, :], aspect='auto', cmap='seismic', vmin=-max_abs,
                   vmax=max_abs)
        plt.colorbar()
        plt.plot(self.explanation['input_data'])
        plt.show()


class TSSaliencyExplainer(TSSaliencyExplainerAIX):
    """
    Wrapper for TSSaliencyExplainer from aix360.
    """

    def __init__(
            self,
            model: Callable,
            input_length: int,
            feature_names: List[str],
            base_value: List[float] = None,
            n_samples: int = 50,
            gradient_samples: int = 25,
            gradient_function: Callable = None,
            random_seed: int = 22,
    ):
        super().__init__(
            model=model,
            input_length=input_length,
            feature_names=feature_names,
            base_value=base_value,
            n_samples=n_samples,
            gradient_samples=gradient_samples,
            gradient_function=gradient_function,
            random_seed=random_seed,
        )

    def explain(self, inputs, outputs=None, **kwargs) -> TSSaliencyResults:
        """
        Explain the model's prediction on X.
        """
        _explanation = super().explain_instance(inputs, y=outputs, **kwargs)
        return TSSaliencyResults(_explanation)
