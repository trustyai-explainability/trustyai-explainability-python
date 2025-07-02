"""Generates visualization according to explanation type"""

# pylint: disable=import-error, wrong-import-order, protected-access, missing-final-newline
from typing import Union, Optional

from bokeh.io import show

from trustyai.explainers import SHAPResults, LimeResults, pdp
from trustyai.metrics.distance import LevenshteinResult
from trustyai.visualizations.visualization_results import VisualizationResults
from trustyai.visualizations.shap import SHAPViz
from trustyai.visualizations.lime import LimeViz
from trustyai.visualizations.pdp import PDPViz
from trustyai.visualizations.distance import DistanceViz


def get_viz(explanations) -> VisualizationResults:
    """
    Get visualization according to the explanation method
    """
    if isinstance(explanations, SHAPResults):
        return SHAPViz()
    if isinstance(explanations, LimeResults):
        return LimeViz()
    if isinstance(explanations, pdp.PDPResults):
        return PDPViz()
    if isinstance(explanations, LevenshteinResult):
        return DistanceViz()
    raise ValueError("Explanation method unknown")


def plot(
    explanations: Union[SHAPResults, LimeResults, pdp.PDPResults, LevenshteinResult],
    output_name: Optional[str] = None,
    render_bokeh: bool = False,
    block: bool = True,
    call_show: bool = True,
) -> None:
    """
    Plot the found feature saliencies.

        Parameters
        ----------
        explanations: Union[LimeResults, SHAPResults, PDPResults, LevenshteinResult]
            the explanation result to plot
        output_name : str
            (default= `None`) The name of the output to be explainer. If `None`, all outputs will
            be displayed
        render_bokeh : bool
            (default= `False`) If true, render plot in bokeh, otherwise use matplotlib.
        block: bool
            (default= `True`) Whether displaying the plot blocks subsequent code execution
        call_show: bool
            (default= 'True') Whether plt.show() will be called by default at the end of the
            plotting function. If `False`, the plot will be returned to the user for further
            editing.
    """
    viz = get_viz(explanations)

    if isinstance(explanations, pdp.PDPResults):
        viz.plot(explanations, output_name)
    elif isinstance(explanations, LevenshteinResult):
        viz.plot(explanations)
    elif output_name is None:
        for output_name_iterator in explanations.saliency_map().keys():
            if render_bokeh:
                show(viz._get_bokeh_plot(explanations, output_name_iterator))
            else:
                viz._matplotlib_plot(
                    explanations, output_name_iterator, block, call_show
                )
    else:
        if render_bokeh:
            show(viz._get_bokeh_plot(explanations, output_name))
        else:
            viz._matplotlib_plot(explanations, output_name, block, call_show)
