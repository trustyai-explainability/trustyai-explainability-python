"""
Wrapper module for TSICEExplainer from aix360.
Original at https://github.com/Trusted-AI/AIX360/
"""

# pylint: disable=too-many-arguments,import-error
from typing import Callable, List, Optional, Union

from aix360.algorithms.tsice import TSICEExplainer as TSICEExplainerAIX
from aix360.algorithms.tsutils.tsperturbers import TSPerturber
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from trustyai.explainers.explanation_results import ExplanationResults


class TSICEResults(ExplanationResults):
    """Wraps TSICE results. This object is returned by the :class:`~TSICEExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, explanation):
        self.explanation = explanation

    def as_dataframe(self) -> pd.DataFrame:
        """Returns the explanation as a pandas dataframe."""
        # Initialize an empty DataFrame
        dataframe = pd.DataFrame()

        # Loop through each feature_name and each key in data_x
        for key in self.explanation["data_x"]:
            for i, feature in enumerate(self.explanation["feature_names"]):
                dataframe[f"{key}-{feature}"] = [
                    val[0] for val in self.explanation["feature_values"][i]
                ]

        # Add "total_impact" as a column
        dataframe["total_impact"] = self.explanation["total_impact"]
        return dataframe

    def as_html(self) -> pd.io.formats.style.Styler:
        """Returns the explanation as an HTML table."""
        dataframe = self.as_dataframe()
        return dataframe.style

    def plot_forecast(self, variable):  # pylint: disable=too-many-locals
        """Plots the explanation.
        Based on https://github.com/Trusted-AI/AIX360/blob/master/examples/tsice/plots.py
        """
        forecast_horizon = self.explanation["current_forecast"].shape[0]
        original_ts = pd.DataFrame(
            data={variable: self.explanation["data_x"][variable]}
        )
        perturbations = [d for d in self.explanation["perturbations"] if variable in d]

        # Generate a list of keys
        keys = list(self.explanation["data_x"].keys())
        # Find the index of the given key
        key = keys.index(variable)
        forecasts_on_perturbations = [
            arr[:, key : key + 1]
            for arr in self.explanation["forecasts_on_perturbations"]
        ]

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

        current_forecast = self.explanation["current_forecast"][:, key : key + 1]
        pred_original_ts = pd.DataFrame(current_forecast, index=new_timestamps)

        _, axis = plt.subplots()

        # Plot perturbed time series
        axis = self._plot_timeseries(
            new_perturbations,
            color="lightgreen",
            axis=axis,
            name="perturbed timeseries samples",
        )

        # Plot original time series
        axis = self._plot_timeseries(
            original_ts, color="green", axis=axis, name="input/original timeseries"
        )

        # Plot varying forecast range
        axis = self._plot_timeseries(
            pred_ts, color="lightblue", axis=axis, name="forecast on perturbed samples"
        )

        # Plot original forecast
        axis = self._plot_timeseries(
            pred_original_ts, color="blue", axis=axis, name="original forecast"
        )

        # Set labels and title
        axis.set_xlabel("Timestamp")
        axis.set_ylabel(variable)
        axis.set_title("Time-Series Individual Conditional Expectation (TSICE)")

        axis.legend()

        # Display the plot
        plt.show()

    def _plot_timeseries(
        self, timeseries, color="green", axis=None, name="time series"
    ):
        showlegend = True
        if isinstance(timeseries, dict):
            data = timeseries
            if isinstance(color, str):
                color = {k: color for k in data}
        elif isinstance(timeseries, list):
            data = {}
            for k, ts_data in enumerate(timeseries):
                data[k] = ts_data
            if isinstance(color, str):
                color = {k: color for k in data}
        else:
            data = {}
            data["default"] = timeseries
            color = {"default": color}

        if axis is None:
            _, axis = plt.subplots()

        first = True
        for key, _timeseries in data.items():
            if not first:
                showlegend = False

            self._add_timeseries(
                axis, _timeseries, color=color[key], showlegend=showlegend, name=name
            )
            first = False

        return axis

    def _add_timeseries(
        self, axis, timeseries, color="green", name="time series", showlegend=False
    ):
        timestamps = timeseries.index
        axis.plot(
            timestamps,
            timeseries[timeseries.columns[0]],
            color=color,
            label=(name if showlegend else "_nolegend_"),
        )

    def plot_impact(self, feature_per_row=2):
        """Plot the impace.
        Based on https://github.com/Trusted-AI/AIX360/blob/master/examples/tsice/plots.py
        """

        n_row = int(np.ceil(len(self.explanation["feature_names"]) / feature_per_row))
        feat_values = np.array(self.explanation["feature_values"])

        fig, axs = plt.subplots(n_row, feature_per_row, figsize=(15, 15))
        axs = axs.ravel()  # Flatten the axs to iterate over it

        for i, feat in enumerate(self.explanation["feature_names"]):
            x_feat = feat_values[i, :, 0]
            trend_fit = LinearRegression()
            trend_line = trend_fit.fit(
                x_feat.reshape(-1, 1), self.explanation["signed_impact"]
            )
            x_trend = np.linspace(min(x_feat), max(x_feat), 101)
            y_trend = trend_line.predict(x_trend[..., np.newaxis])

            # Scatter plot
            axs[i].scatter(x=x_feat, y=self.explanation["signed_impact"], color="blue")
            # Line plot
            axs[i].plot(
                x_trend,
                y_trend,
                color="green",
                label="correlation between forecast and observed feature",
            )
            # Reference line
            current_value = self.explanation["current_feature_values"][i][0]
            axs[i].axvline(
                x=current_value,
                color="firebrick",
                linestyle="--",
                label="current value",
            )

            axs[i].set_xlabel(feat)
            axs[i].set_ylabel("Δ forecast")

        # Display the legend on the first subplot
        axs[0].legend()

        fig.suptitle("Impact of Derived Variable On The Forecast", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()


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
