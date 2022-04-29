"""Visualiser utilies for explainer results"""
from abc import ABC, abstractmethod
import pandas as pd
from pandas.io.formats.style import Styler

# pylint: disable=too-few-public-methods
class ExplanationVisualiser(ABC):
    """Abstract class for explanation visualisers"""

    @abstractmethod
    def as_dataframe(self) -> pd.DataFrame:
        """Display explanation result as a dataframe"""

    @abstractmethod
    def as_html(self) -> Styler:
        """Visualise the styled dataframe"""


DEFAULT_STYLE = {
    "positive_primary_colour": "#13ba3c",
    "negative_primary_colour": "#ee0000",
    "neutral_primary_colour": "#ffffff",
}


DEFAULT_RC_PARAMS = {
    "patch.linewidth": 0.5,
    "patch.facecolor": "348ABD",
    "patch.edgecolor": "EEEEEE",
    "patch.antialiased": True,
    "font.size": 10.0,
    "axes.facecolor": "DDDDDD",
    "axes.edgecolor": "white",
    "axes.linewidth": 1,
    "axes.grid": True,
    "axes.titlesize": "x-large",
    "axes.labelsize": "large",
    "axes.labelcolor": "black",
    "axes.axisbelow": True,
    "text.color": "black",
    "xtick.color": "black",
    "xtick.direction": "out",
    "ytick.color": "black",
    "ytick.direction": "out",
    "legend.facecolor": "ffffff",
    "grid.color": "white",
    "grid.linestyle": "-",  # solid line
    "figure.figsize": (16, 9),
    "figure.dpi": 100,
    "figure.facecolor": "ffffff",
    "figure.edgecolor": "777777",
    "savefig.bbox": "tight",
}
