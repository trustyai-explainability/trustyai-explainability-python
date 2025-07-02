"""Visualizations.pdp module"""

# pylint: disable = import-error, wrong-import-order, too-few-public-methods, missing-final-newline
# pylint: disable = protected-access
import matplotlib.pyplot as plt

from trustyai.explainers.pdp import PDPResults


class PDPViz:
    """Visualizes PDP graphs"""

    def plot(self, explanations, output_name=None, block=True, call_show=True) -> None:
        """
        Parameters
        ----------
        explanations: pdp.PDPResults
            the partial dependence plots associated to the model outputs
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
        pdp_graphs = explanations.pdp_graphs
        fig, axs = plt.subplots(len(pdp_graphs), constrained_layout=True)
        p_idx = 0
        for pdp_graph in pdp_graphs:
            if output_name is not None and output_name != str(
                pdp_graph.getOutput().getName()
            ):
                continue
            fig.suptitle(str(pdp_graph.getOutput().getName()))
            pdp_x = []
            for i in range(len(pdp_graph.getX())):
                pdp_x.append(PDPResults._to_plottable(pdp_graph.getX()[i]))
            pdp_y = []
            for i in range(len(pdp_graph.getY())):
                pdp_y.append(PDPResults._to_plottable(pdp_graph.getY()[i]))
            axs[p_idx].plot(pdp_x, pdp_y)
            axs[p_idx].set_title(
                str(pdp_graph.getFeature().getName()), loc="left", fontsize="small"
            )
            axs[p_idx].grid()
            p_idx += 1
        fig.supylabel("Partial Dependence Plot")
        if call_show:
            plt.show(block=block)
