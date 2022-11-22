from abc import ABC, abstractmethod
from typing import Dict

import bokeh.models
import pandas as pd
from bokeh.io import show
from pandas.io.formats.style import Styler


class ExplanationResults(ABC):
    """Abstract class for explanation visualisers"""
    @abstractmethod
    def as_dataframe(self) -> pd.DataFrame:
        """Display explanation result as a dataframe"""

    @abstractmethod
    def as_html(self) -> Styler:
        """Visualise the styled dataframe"""


# pylint: disable=too-few-public-methods
class SaliencyResults(ExplanationResults):
    """Abstract class for saliency visualisers"""
    @abstractmethod
    def saliency_map(self):
        """Return the Saliencies as a dictionary, keyed by output name"""

    @abstractmethod
    def _matplotlib_plot(self, output_name: str) -> None:
        """Plot the saliencies of a particular output in matplotlib"""

    @abstractmethod
    def _get_bokeh_plot(self, output_name: str) -> bokeh.models.Plot:
        """Get a bokeh plot visualizing the saliencies of a particular output"""

    def _get_bokeh_plot_dict(self) -> Dict[str, bokeh.models.Plot]:
        """Get a dictionary containing visualizations of the saliencies of all outputs,
        keyed by output name"""
        return {output_name: self._get_bokeh_plot(output_name)
                for output_name in self.saliency_map().keys()}

    def plot(self, output_name=None, bokeh=False) -> None:
        """
        Plot the saliencies of a particular output

        Parameters
        ----------
        output_name : str
            (default=None) The name of the output to be explainer. If `None`, all outputs will
             be displayed
        bokeh : bool
            (default: false) Whether to render as bokeh (true) or matplotlib (false)
        """
        if output_name is None:
            for output_name_iterator in self.saliency_map().keys():
                if bokeh:
                    show(self._get_bokeh_plot(output_name_iterator))
                else:
                    self._matplotlib_plot(output_name_iterator)
        else:
            if bokeh:
                show(self._get_bokeh_plot(output_name))
            else:
                self._matplotlib_plot(output_name)