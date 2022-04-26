"""Visualiser utilies for explainer results"""
from abc import ABC, abstractmethod
import pandas as pd


class ExplanationVisualiser(ABC):
    @abstractmethod
    def as_dataframe(self) -> pd.DataFrame:
        """Display explanation result as a dataframe"""
        pass
