"""Explainers.pdp module"""

import math
import matplotlib.pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler

from jpype import (
    JImplements,
    JOverride,
)

# pylint: disable = import-error
from org.kie.trustyai.explainability.global_ import pdp

# pylint: disable = import-error
from org.kie.trustyai.explainability.model import (
    PredictionProvider,
    PredictionInputsDataDistribution,
    PredictionOutput,
    Output,
    Type,
    Value,
)

from trustyai.utils.data_conversions import ManyInputsUnionType, many_inputs_convert

from .explanation_results import ExplanationResults


class PDPResults(ExplanationResults):
    """
    Results class for Partial Dependence Plots
    """

    def __init__(self, pdp_graphs):
        self.pdp_graphs = pdp_graphs

    def as_dataframe(self) -> pd.DataFrame:
        """
        Returns
        -------
        a pd.DataFrame with input values and feature name as
        columns and marginal feature outputs as rows
        """
        pdp_series_list = []
        for pdp_graph in self.pdp_graphs:
            inputs = [self._to_plottable(x) for x in pdp_graph.getX()]
            outputs = [self._to_plottable(y) for y in pdp_graph.getY()]
            pdp_dict = dict(zip(inputs, outputs))
            pdp_dict["feature"] = "" + str(pdp_graph.getFeature().getName())
            pdp_series = pd.Series(index=inputs + ["feature"], data=pdp_dict)
            pdp_series_list.append(pdp_series)
        pdp_df = pd.DataFrame(pdp_series_list)
        return pdp_df

    def as_html(self) -> Styler:
        """
        Returns
        -------
        Style object from the PDP pd.DataFrame (see as_dataframe)
        """
        return self.as_dataframe().style

    def plot(self, output_name=None, block=True, call_show=True) -> None:
        """
        Parameters
        ----------
        output_name: str
            name of the output to be plotted
            Default to None
        block: bool
            whether the plotting operation
            should be blocking or not
        call_show: bool
            (default= 'True') Whether plt.show() will be called by default at the end of
            the plotting function. If `False`, the plot will be returned to the user for
            further editing.
        """
        fig, axs = plt.subplots(len(self.pdp_graphs), constrained_layout=True)
        p_idx = 0
        for pdp_graph in self.pdp_graphs:
            if output_name is not None and output_name != str(
                pdp_graph.getOutput().getName()
            ):
                continue
            fig.suptitle(str(pdp_graph.getOutput().getName()))
            pdp_x = []
            for i in range(len(pdp_graph.getX())):
                pdp_x.append(self._to_plottable(pdp_graph.getX()[i]))
            pdp_y = []
            for i in range(len(pdp_graph.getY())):
                pdp_y.append(self._to_plottable(pdp_graph.getY()[i]))
            axs[p_idx].plot(pdp_x, pdp_y)
            axs[p_idx].set_title(
                str(pdp_graph.getFeature().getName()), loc="left", fontsize="small"
            )
            axs[p_idx].grid()
            p_idx += 1
        fig.supylabel("Partial Dependence Plot")
        if call_show:
            plt.show(block=block)

    @staticmethod
    def _to_plottable(datum: Value):
        plottable = datum.asNumber()
        if math.isnan(plottable):
            plottable = str(datum.asString())
        return plottable


# pylint: disable = too-few-public-methods
class PDPExplainer:
    """
    Partial Dependence Plot explainer.
    See https://christophm.github.io/interpretable-ml-book/pdp.html
    """

    def __init__(self, config=None):
        if config is None:
            config = pdp.PartialDependencePlotConfig()
        self._explainer = pdp.PartialDependencePlotExplainer(config)

    def explain(
        self, model: PredictionProvider, data: ManyInputsUnionType, num_outputs: int = 1
    ) -> PDPResults:
        """
        Parameters
        ----------
        model: PredictionProvider
            the model to explain
        data: ManyInputsUnionType
            the data used to calculate the PDP
        num_outputs: int
            the number of outputs to calculate the PDP for

        Returns
        -------
        pdp_results: PDPResults
            the partial dependence plots associated to the model outputs
        """
        metadata = _PredictionProviderMetadata(many_inputs_convert(data), num_outputs)
        pdp_graphs = self._explainer.explainFromMetadata(model, metadata)
        return PDPResults(pdp_graphs)


@JImplements(
    "org.kie.trustyai.explainability.model.PredictionProviderMetadata", deferred=True
)
class _PredictionProviderMetadata:
    """
    Implementation of org.kie.trustyai.explainability.model.PredictionProviderMetadata interface
    """

    def __init__(self, data: list, size: int):
        """
        Parameters
        ----------
        data: ManyInputsUnionType
            the data
        size: int
            the size of the model output
        """
        self.data = PredictionInputsDataDistribution(data)
        outputs = []
        for _ in range(size):
            outputs.append(Output("", Type.UNDEFINED))
        self.pred_out = PredictionOutput(outputs)

    # pylint: disable = invalid-name
    @JOverride
    def getDataDistribution(self):
        """
        Returns
        --------
        the underlying data distribution
        """
        return self.data

    # pylint: disable = invalid-name
    @JOverride
    def getInputShape(self):
        """
        Returns
        --------
        a PredictionInput from the underlying distribution
        """
        return self.data.sample()

    # pylint: disable = invalid-name
    @JOverride
    def getOutputShape(self):
        """
        Returns
        --------
        a PredictionOutput
        """
        return self.pred_out
