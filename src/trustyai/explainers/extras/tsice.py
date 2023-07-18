"""
Wrapper module for TSICEExplainer from aix360.
Original at https://github.com/Trusted-AI/AIX360/
"""
# pylint: disable=too-many-arguments,import-error
from typing import Callable, List, Optional, Union

from aix360.algorithms.tsice import TSICEExplainer as TSICEExplainerAIX
from aix360.algorithms.tsutils.tsperturbers import TSPerturber
import bokeh
import pandas as pd

from trustyai.model import SaliencyResults


class TSICEResults(SaliencyResults):
    """Wraps TSICE results. This object is returned by the :class:`~TSICEExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, explanation):
        self.explanation = explanation

    def as_dataframe(self) -> pd.DataFrame:
        """Returns the explanation as a pandas dataframe."""
        return pd.DataFrame(self.explanation)

    def as_html(self) -> pd.io.formats.style.Styler:
        """Returns the explanation as an HTML table."""
        dataframe = self.as_dataframe()
        return dataframe.style

    def saliency_map(self):
        """
        Returns a dictionary of feature names and their total impact.
        """
        dict(zip(self.explanation["feature_names"], self.explanation["total_impact"]))

    def _matplotlib_plot(self, output_name: str, block: bool, call_show: bool) -> None:
        pass

    def _get_bokeh_plot(self, output_name: str) -> bokeh.models.Plot:
        pass


class TSICEExplainer(TSICEExplainerAIX):
    """
    Wrapper for TSICEExplainer from aix360.
    """

    def __init__(
        self,
        model: Callable,
        input_length: int,
        forecast_lookahead: int,
        n_variables: int = 1,
        n_exogs: int = 0,
        n_perturbations: int = 25,
        features_to_analyze: Optional[List[str]] = None,
        perturbers: Optional[List[Union[TSPerturber, dict]]] = None,
        explanation_window_start: Optional[int] = None,
        explanation_window_length: int = 10,
    ):
        super().__init__(
            forecaster=model,
            input_length=input_length,
            forecast_lookahead=forecast_lookahead,
            n_variables=n_variables,
            n_exogs=n_exogs,
            n_perturbations=n_perturbations,
            features_to_analyze=features_to_analyze,
            perturbers=perturbers,
            explanation_window_start=explanation_window_start,
            explanation_window_length=explanation_window_length,
        )

    def explain(self, inputs, outputs=None, **kwargs) -> TSICEResults:
        """
        Explain the model's prediction on X.
        """
        _explanation = super().explain_instance(inputs, y=outputs, **kwargs)
        return TSICEResults(_explanation)
