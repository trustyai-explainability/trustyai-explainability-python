"""
Wrapper module for TSICEExplainer from aix360.
Original at https://github.com/Trusted-AI/AIX360/
"""
# pylint: disable=too-many-arguments,import-error
from typing import Callable, List, Optional, Union

from aix360.algorithms.tsice import TSICEExplainer as TSICEExplainerAIX
from aix360.algorithms.tsutils.tsperturbers import TSPerturber
import bokeh
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from trustyai.explainers.explanation_results import ExplanationResults

import matplotlib.pyplot as plt
from typing import Union
import pandas as pd


class TSICEResults:
    """Wraps TSICE results. This object is returned by the :class:`~TSICEExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, explanation):
        self.explanation = explanation

    def as_dataframe(self) -> pd.DataFrame:
        """Returns the explanation as a pandas dataframe."""
        # Initialize an empty DataFrame
        df = pd.DataFrame()

        # Loop through each feature_name and each key in data_x
        for key in self.explanation["data_x"]:
            for i, feature in enumerate(self.explanation["feature_names"]):
                df[f"{key}-{feature}"] = [val[0] for val in self.explanation["feature_values"][i]]

        # Add "total_impact" as a column
        df["total_impact"] = self.explanation["total_impact"]
        return df

    def as_html(self) -> pd.io.formats.style.Styler:
        """Returns the explanation as an HTML table."""
        dataframe = self.as_dataframe()
        return dataframe.style

    def plot_forecast(self, variable):
        """Plots the explanation.
        Based on https://github.com/Trusted-AI/AIX360/blob/master/examples/tsice/plots.py"""
        forecast_horizon = self.explanation['current_forecast'].shape[0]
        original_ts = pd.DataFrame(data={variable: self.explanation["data_x"][variable]})
        perturbations = [d for d in self.explanation['perturbations'] if variable in d]

        # Generate a list of keys
        keys = list(self.explanation["data_x"].keys())
        # Find the index of the given key
        key = keys.index(variable)
        forecasts_on_perturbations = [arr[:, key:key+1] for arr in self.explanation["forecasts_on_perturbations"]]

        new_perturbations = []
        new_timestamps = []
        pred_ts = []

        original_ts.index.freq = pd.infer_freq(original_ts.index)
        for i in range(1, forecast_horizon + 1):
            new_timestamps.append(original_ts.index[-1] + (i * original_ts.index.freq))

        for perturbation in perturbations:
            new_perturbations.append(pd.DataFrame(perturbation))

        for forecast in forecasts_on_perturbations:
            pred_ts.append(pd.DataFrame(forecast, index=new_timestamps))

        current_forecast = self.explanation["current_forecast"][:, key:key+1]
        pred_original_ts = pd.DataFrame(
            current_forecast, index=new_timestamps
        )

        fig, ax = plt.subplots()

        # Plot perturbed time series
        ax = self._plot_timeseries(new_perturbations, color="lightgreen", ax=ax, name="perturbed timeseries samples")

        # Plot original time series
        ax = self._plot_timeseries(original_ts, color="green", ax=ax, name="input/original timeseries")

        # Plot varying forecast range
        ax = self._plot_timeseries(pred_ts, color="lightblue", ax=ax, name="forecast on perturbed samples")

        # Plot original forecast
        ax = self._plot_timeseries(pred_original_ts, color="blue", ax=ax, name="original forecast")

        # Set labels and title
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(variable)
        ax.set_title("Time-Series Individual Conditional Expectation (TSICE)")

        ax.legend()

        # Display the plot
        plt.show()

        # Return the figure
        return fig

    def _plot_timeseries(self, ts, color="green", ax=None, name="time series"):
        showlegend = True
        if type(ts) == dict:
            data = ts
            if type(color) == str:
                color = {k: color for k in data}
        elif type(ts) == list:
            data = {}
            for k, ts_data in enumerate(ts):
                data[k] = ts_data
            if type(color) == str:
                color = {k: color for k in data}
        else:
            data = {}
            data["default"] = ts
            color = {"default": color}

        if ax is None:
            fig, ax = plt.subplots()

        first = True
        for key, ts in data.items():
            if not first:
                showlegend = False

            self._add_timeseries(ax, ts, color=color[key], showlegend=showlegend, name=name)
            first = False

        return ax

    def _add_timeseries(self, ax, ts, color="green", name="time series", showlegend=False):
        timestamps = ts.index
        ax.plot(timestamps, ts[ts.columns[0]], color=color, label=(name if showlegend else '_nolegend_'))


    def plot_impact(self, feature_per_row=2):
        """Plot the impace.
        Based on https://github.com/Trusted-AI/AIX360/blob/master/examples/tsice/plots.py"""

        df = pd.DataFrame(self.explanation["data_x"])
        n_row = int(np.ceil(len(self.explanation["feature_names"]) / feature_per_row))
        feat_values = np.array(self.explanation["feature_values"])

        fig, axs = plt.subplots(n_row, feature_per_row, figsize=(15, 15))
        axs = axs.ravel()  # Flatten the axs to iterate over it

        for i, feat in enumerate(self.explanation["feature_names"]):
            x_feat = feat_values[i, :, 0]
            trend_fit = LinearRegression()
            trend_line = trend_fit.fit(x_feat.reshape(-1, 1), self.explanation["signed_impact"])
            x_trend = np.linspace(min(x_feat), max(x_feat), 101)
            y_trend = trend_line.predict(x_trend[..., np.newaxis])

            # Scatter plot
            axs[i].scatter(x=x_feat, y=self.explanation["signed_impact"], color='blue')
            # Line plot
            axs[i].plot(x_trend, y_trend, color="green", label="correlation between forecast and observed feature")
            # Reference line
            current_value = self.explanation["current_feature_values"][i][0]
            axs[i].axvline(x=current_value, color='firebrick', linestyle='--', label="current value")

            axs[i].set_xlabel(feat)
            axs[i].set_ylabel('Δ forecast')

        # Display the legend on the first subplot
        axs[0].legend()

        fig.suptitle("Impact of Derived Variable On The Forecast", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()
        return fig


class TSICEExplainer(TSICEExplainerAIX):
    """
    Wrapper for TSICEExplainer from aix360.
    """

    def __init__(
            self,
            model: Callable,
            input_length: int,
            forecast_lookahead: int,
            n_variables: int = 1,
            n_exogs: int = 0,
            n_perturbations: int = 25,
            features_to_analyze: Optional[List[str]] = None,
            perturbers: Optional[List[Union[TSPerturber, dict]]] = None,
            explanation_window_start: Optional[int] = None,
            explanation_window_length: int = 10,
    ):
        super().__init__(
            forecaster=model,
            input_length=input_length,
            forecast_lookahead=forecast_lookahead,
            n_variables=n_variables,
            n_exogs=n_exogs,
            n_perturbations=n_perturbations,
            features_to_analyze=features_to_analyze,
            perturbers=perturbers,
            explanation_window_start=explanation_window_start,
            explanation_window_length=explanation_window_length,
        )

    def explain(self, inputs, outputs=None, **kwargs) -> TSICEResults:
        """
        Explain the model's prediction on X.
        """
        _explanation = super().explain_instance(inputs, y=outputs, **kwargs)
        return TSICEResults(_explanation)
