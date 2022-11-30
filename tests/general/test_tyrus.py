import pytest


from trustyai.model import Model
from trustyai.utils.tyrus import Tyrus
import numpy as np
import pandas as pd

import os


def test_tyrus_series():
    # define data
    data = pd.DataFrame(np.random.rand(101, 5), columns=list('ABCDE'))

    # define model
    def predict_function(x):
        return pd.DataFrame(
            np.stack(
                [x.sum(1), x.std(1), np.linalg.norm(x, axis=1)]).T,
            columns= ['Sum', 'StdDev', 'L2 Norm'])

    predictions = predict_function(data)

    model = Model(predict_function, dataframe_input=True)

    # create Tyrus instance
    tyrus = Tyrus(
        model,
        data.iloc[100],
        predictions.iloc[100],
        background=data.iloc[:100]
    )

    # launch dashboard
    tyrus.run()

    # see if dashboard html exists
    assert "trustyai_dashboard.html" in os.listdir()

    # cleanup
    os.remove("trustyai_dashboard.html")


def test_tyrus_numpy():
    # define data
    data = np.random.rand(101, 5)

    # define model
    def predict_function(x):
        return np.stack([x.sum(1), x.std(1), np.linalg.norm(x, axis=1)]).T

    predictions = predict_function(data)

    model = Model(predict_function, dataframe_input=False)

    # create Tyrus instance
    tyrus = Tyrus(
        model,
        data[100],
        predictions[100],
        background=data[:100]
    )

    # launch dashboard
    tyrus.run()

    # see if dashboard html exists
    assert "trustyai_dashboard.html" in os.listdir()

    # cleanup
    os.remove("trustyai_dashboard.html")
