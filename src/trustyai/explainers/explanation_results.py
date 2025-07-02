"""Generic class for Explanation and Saliency results"""

from abc import ABC, abstractmethod

import pandas as pd
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
