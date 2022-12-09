"""Explainers.pdp module"""

from typing import Dict, Union

import bokeh.models
import matplotlib.pyplot as plt
import matplotlib as mpl
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import pandas as pd
import numpy as np
from pandas.io.formats.style import Styler

from trustyai.utils._visualisation import (
    DEFAULT_STYLE as ds,
    DEFAULT_RC_PARAMS as drcp,
    bold_red_html,
    bold_green_html,
    output_html,
    feature_html,
)

from trustyai.utils.data_conversions import (
    OneInputUnionType,
    data_conversion_docstring,
    OneOutputUnionType,
)
from trustyai.utils.data_conversions import (
    ManyInputsUnionType
)

from jpype import (
    JImplements,
    JOverride,
)

from org.kie.trustyai.explainability.global_ import pdp

from .explanation_results import ExplanationResults
from trustyai.model import simple_prediction, Model

from org.kie.trustyai.explainability.model import (
    PredictionProvider,
    PredictionInputsDataDistribution,
    PredictionOutput,
    Output,
    Type
)

from java.util import Random


class PDPExplainer:

    def __init__(self, config=None):
        if config is None:
            config = pdp.PartialDependencePlotConfig()
        self._explainer = pdp.PartialDependencePlotExplainer(config)


    def explain(self, model: PredictionProvider, data: ManyInputsUnionType, num_outputs: int = 1):
        """
        Parameters
        ----------


        Returns
        -------

        """
        pdp_graphs = self._explainer.explainFromMetadata(model, PredictionProviderMetadata(data, num_outputs))
        return PDPResults(pdp_graphs)


class PDPResults(ExplanationResults):

    def __init__(self, pdp_graphs):
        self.pdp_graphs = pdp_graphs

    def as_dataframe(self) -> pd.DataFrame:
        pdp_series_list = []
        for pdp_graph in self.pdp_graphs:
            inputs = [x.getUnderlyingObject() for x in pdp_graph.getX()]
            outputs = [y.getUnderlyingObject() for y in pdp_graph.getY()]
            pdp_dict = dict(zip(inputs,
                                outputs))
            pdp_dict['feature'] = '' + str(pdp_graph.getFeature().getName())
            pdp_series = pd.Series(index=inputs + ['feature'], data=pdp_dict)
            pdp_series_list.append(pdp_series)
        pdp_df = pd.DataFrame(pdp_series_list)
        return pdp_df

    def as_html(self) -> Styler:
        return self.as_dataframe().style

    def _matplotlib_plot(self, output_name, block=True) -> None:
        fig, axs = plt.subplots(len(self.pdp_graphs), sharex=True)
        p = 0
        for pdp_graph in self.pdp_graphs:
            if output_name is not None and output_name != str(pdp_graph.getOutput().getName()):
                continue
            fig.suptitle(str(pdp_graph.getOutput().getName()))
            pdp_data = []
            for i in range(len(pdp_graph.getX())):
                pdp_data.append([pdp_graph.getX()[i].getUnderlyingObject(), pdp_graph.getY()[i].getUnderlyingObject()])
            axs[p].plot(np.array(pdp_data))
            p += 1
        plt.show(block=block)


@JImplements("org.kie.trustyai.explainability.model.PredictionProviderMetadata", deferred=True)
class PredictionProviderMetadata:
    """
    Wraps java LocalExplainer interface, delegating the generation of explanations
    to either LimeExplainer or SHAPExplainer.
    """

    def __init__(self, data: ManyInputsUnionType, size: int):
        """
        Parameters
        ----------
        wrapped: Union[trustyai.explainer.LimeExplainer, trustyai.explainer.SHAPExplainer]
            wrapped explainer
        """
        self.data = PredictionInputsDataDistribution(data)
        outputs = []
        for i in range(size):
            outputs.append(Output("", Type.UNDEFINED))
        self.pred_out = PredictionOutput(outputs)

    @JOverride
    def getDataDistribution(self):
        """
        Returns
        --------

        """
        return self.data

    @JOverride
    def getInputShape(self):
        """
        Returns
        --------

        """
        return self.data.sample()

    @JOverride
    def getOutputShape(self):
        """
        Returns
        --------

        """
        return self.pred_out
