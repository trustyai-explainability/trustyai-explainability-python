"""Visualizations.shap module"""

# pylint: disable = import-error, consider-using-f-string, too-few-public-methods, missing-final-newline
import matplotlib.pyplot as plt
import matplotlib as mpl
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import pandas as pd
import numpy as np

from trustyai.utils._visualisation import (
    DEFAULT_STYLE as ds,
    DEFAULT_RC_PARAMS as drcp,
    bold_red_html,
    bold_green_html,
    output_html,
    feature_html,
)
from trustyai.visualizations.visualization_results import VisualizationResults


class SHAPViz(VisualizationResults):
    """Visualizes SHAP results."""

    def _matplotlib_plot(
        self, explanations, output_name=None, block=True, call_show=True
    ) -> None:
        """Visualize the SHAP explanation of each output as a set of candlestick plots,
        one per output."""
        with mpl.rc_context(drcp):
            shap_values = [
                pfi.getScore()
                for pfi in explanations.saliency_map()[
                    output_name
                ].getPerFeatureImportance()[:-1]
            ]
            feature_names = [
                str(pfi.getFeature().getName())
                for pfi in explanations.saliency_map()[
                    output_name
                ].getPerFeatureImportance()[:-1]
            ]
            fnull = explanations.get_fnull()[output_name]
            prediction = fnull + sum(shap_values)

            if call_show:
                plt.figure()
            pos = fnull
            for j, shap_value in enumerate(shap_values):
                color = (
                    ds["negative_primary_colour"]
                    if shap_value < 0
                    else ds["positive_primary_colour"]
                )
                width = 0.9
                if j > 0:
                    plt.plot([j - 0.5, j + width / 2 * 0.99], [pos, pos], color=color)
                plt.bar(j, height=shap_value, bottom=pos, color=color, width=width)
                pos += shap_values[j]

                if j != len(shap_values) - 1:
                    plt.plot([j - width / 2 * 0.99, j + 0.5], [pos, pos], color=color)

            plt.axhline(
                fnull,
                color="#444444",
                linestyle="--",
                zorder=0,
                label="Background Value",
            )
            plt.axhline(prediction, color="#444444", zorder=0, label="Prediction")
            plt.legend()

            ticksize = np.diff(plt.gca().get_yticks())[0]
            plt.ylim(
                plt.gca().get_ylim()[0] - ticksize / 2,
                plt.gca().get_ylim()[1] + ticksize / 2,
            )
            plt.xticks(np.arange(len(feature_names)), feature_names)
            plt.ylabel(explanations.saliency_map()[output_name].getOutput().getName())
            plt.xlabel("Feature SHAP Value")
            plt.title(f"SHAP: Feature Contributions to {output_name}")
            if call_show:
                plt.show(block=block)

    def _get_bokeh_plot(self, explanations, output_name):
        fnull = explanations.get_fnull()[output_name]

        # create dataframe of plot values
        data_source = pd.DataFrame(
            [
                {
                    "feature": str(pfi.getFeature().getName()),
                    "saliency": pfi.getScore(),
                }
                for pfi in explanations.saliency_map()[
                    output_name
                ].getPerFeatureImportance()[:-1]
            ]
        )
        prediction = fnull + data_source["saliency"].sum()

        data_source["color"] = data_source["saliency"].apply(
            lambda x: (
                ds["positive_primary_colour"]
                if x >= 0
                else ds["negative_primary_colour"]
            )
        )
        data_source["color_faded"] = data_source["saliency"].apply(
            lambda x: (
                ds["positive_primary_colour_faded"]
                if x >= 0
                else ds["negative_primary_colour_faded"]
            )
        )
        data_source["index"] = data_source.index
        data_source["saliency_text"] = data_source["saliency"].apply(
            lambda x: (bold_red_html if x <= 0 else bold_green_html)("{:.2f}".format(x))
        )
        data_source["bottom"] = pd.Series(
            [fnull] + data_source["saliency"].iloc[0:-1].tolist()
        ).cumsum()
        data_source["top"] = data_source["bottom"] + data_source["saliency"]

        # create hovertools
        htool_fnull = HoverTool(
            name="fnull",
            tooltips=("<h3>SHAP</h3>Baseline {}: {}").format(
                output_name, output_html("{:.2f}".format(fnull))
            ),
            line_policy="interp",
        )
        htool_pred = HoverTool(
            name="pred",
            tooltips=("<h3>SHAP</h3>Predicted {}: {}").format(
                output_name, output_html("{:.2f}".format(prediction))
            ),
            line_policy="interp",
        )
        htool_bars = HoverTool(
            name="bars",
            tooltips="<h3>SHAP</h3> {} contributions to {}: @saliency_text".format(
                feature_html("@feature"), output_html(output_name)
            ),
        )

        # create plot
        bokeh_plot = figure(
            sizing_mode="stretch_both",
            title="SHAP Feature Contributions",
            x_range=data_source["feature"],
            tools=[htool_pred, htool_fnull, htool_bars],
        )

        # add fnull and background lines
        line_data_source = ColumnDataSource(
            pd.DataFrame(
                [
                    {"x": 0, "pred": prediction},
                    {"x": len(data_source), "pred": prediction},
                ]
            )
        )
        fnull_data_source = ColumnDataSource(
            pd.DataFrame(
                [{"x": 0, "fnull": fnull}, {"x": len(data_source), "fnull": fnull}]
            )
        )

        bokeh_plot.line(
            x="x",
            y="fnull",
            line_color="#999",
            hover_line_color="#333",
            line_width=2,
            hover_line_width=4,
            line_dash="dashed",
            name="fnull",
            source=fnull_data_source,
        )
        bokeh_plot.line(
            x="x",
            y="pred",
            line_color="#999",
            hover_line_color="#333",
            line_width=2,
            hover_line_width=4,
            name="pred",
            source=line_data_source,
        )

        # create candlestick plot lines
        bokeh_plot.line(
            x=[0.5, 1],
            y=data_source.iloc[0]["top"],
            color=data_source.iloc[0]["color"],
        )
        for i in range(1, len(data_source)):
            # bar left line
            bokeh_plot.line(
                x=[i, i + 0.5],
                y=data_source.iloc[i]["bottom"],
                color=data_source.iloc[i]["color"],
            )
            # bar right line
            if i != len(data_source) - 1:
                bokeh_plot.line(
                    x=[i + 0.5, i + 1],
                    y=data_source.iloc[i]["top"],
                    color=data_source.iloc[i]["color"],
                )

        # create candles
        bokeh_plot.vbar(
            x="feature",
            bottom="bottom",
            top="top",
            hover_color="color",
            color="color_faded",
            width=0.75,
            name="bars",
            source=data_source,
        )
        bokeh_plot.yaxis.axis_label = str(output_name)
        return bokeh_plot

    def _get_bokeh_plot_dict(self, explanations):
        return {
            decision: self._get_bokeh_plot(explanations, decision)
            for decision in explanations.saliency_map().keys()
        }
