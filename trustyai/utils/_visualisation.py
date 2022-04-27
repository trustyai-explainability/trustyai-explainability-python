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
