"""Visualizations.lime module"""

# pylint: disable = import-error, too-few-public-methods, consider-using-f-string, missing-final-newline
import matplotlib.pyplot as plt
import matplotlib as mpl
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import pandas as pd

from trustyai.utils._visualisation import (
    DEFAULT_STYLE as ds,
    DEFAULT_RC_PARAMS as drcp,
    bold_red_html,
    bold_green_html,
    output_html,
    feature_html,
)
from trustyai.visualizations.visualization_results import VisualizationResults


class LimeViz(VisualizationResults):
    """Visualizes LIME results."""

    def _matplotlib_plot(
        self, explanations, output_name: str, block=True, call_show=True
    ) -> None:
        """Plot the LIME saliencies."""
        with mpl.rc_context(drcp):
            dictionary = {}
            for feature_importance in (
                explanations.saliency_map().get(output_name).getPerFeatureImportance()
            ):
                dictionary[
                    feature_importance.getFeature().name
                ] = feature_importance.getScore()

            colours = [
                (
                    ds["negative_primary_colour"]
                    if i < 0
                    else ds["positive_primary_colour"]
                )
                for i in dictionary.values()
            ]
            plt.title(f"LIME: Feature Importances to {output_name}")
            plt.barh(
                range(len(dictionary)),
                dictionary.values(),
                align="center",
                color=colours,
            )
            plt.yticks(range(len(dictionary)), list(dictionary.keys()))
            plt.tight_layout()

            if call_show:
                plt.show(block=block)

    def _get_bokeh_plot(self, explanations, output_name):
        lime_data_source = pd.DataFrame(
            [
                {
                    "feature": str(pfi.getFeature().getName()),
                    "saliency": pfi.getScore(),
                }
                for pfi in explanations.saliency_map()[
                    output_name
                ].getPerFeatureImportance()
            ]
        )
        lime_data_source["color"] = lime_data_source["saliency"].apply(
            lambda x: (
                ds["positive_primary_colour"]
                if x >= 0
                else ds["negative_primary_colour"]
            )
        )
        lime_data_source["saliency_colored"] = lime_data_source["saliency"].apply(
            lambda x: (bold_green_html if x >= 0 else bold_red_html)("{:.2f}".format(x))
        )

        lime_data_source["color_faded"] = lime_data_source["saliency"].apply(
            lambda x: (
                ds["positive_primary_colour_faded"]
                if x >= 0
                else ds["negative_primary_colour_faded"]
            )
        )
        source = ColumnDataSource(lime_data_source)
        htool = HoverTool(
            name="bars",
            tooltips="<h3>LIME</h3> {} saliency to {}: @saliency_colored".format(
                feature_html("@feature"), output_html(output_name)
            ),
        )
        bokeh_plot = figure(
            sizing_mode="stretch_both",
            title="Lime Feature Importances",
            y_range=lime_data_source["feature"],
            tools=[htool],
        )
        bokeh_plot.hbar(
            y="feature",
            left=0,
            right="saliency",
            fill_color="color_faded",
            line_color="color",
            hover_color="color",
            color="color",
            height=0.75,
            name="bars",
            source=source,
        )
        bokeh_plot.line([0, 0], [0, len(lime_data_source)], color="#000")
        bokeh_plot.xaxis.axis_label = "Saliency Value"
        bokeh_plot.yaxis.axis_label = "Feature"
        return bokeh_plot

    def _get_bokeh_plot_dict(self, explanations):
        return {
            output_name: self._get_bokeh_plot(explanations, output_name)
            for output_name in explanations.saliency_map().keys()
        }
