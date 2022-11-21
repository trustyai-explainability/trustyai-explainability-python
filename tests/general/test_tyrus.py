import pytest


from trustyai.model import Model
from trustyai.utils.tyrus import Tyrus
import numpy as np
import pandas as pd

import os


def test_tyrus():
    # define data
    data = pd.DataFrame(np.random.rand(101, 5), columns=list('ABCDE'))

    # define model
    def predict_function(x):
        return pd.DataFrame(
            np.stack(
                [x.sum(1), x.std(1), np.linalg.norm(x, axis=1)]).T,
            columns= ['Sum', 'StdDev', 'L2 Norm'])

    predictions = predict_function(data)

    model = Model(predict_function, dataframe_input=True, arrow=True)
    original_input = data.iloc[100:]
    original_output = predictions.iloc[100:]

    # create Tyrus instance
    tyrus = Tyrus(
        model,
        original_input,
        original_output,
        background=data.iloc[:100]
    )

    # launch dashboard
    tyrus.run()

    # see if dashboard html exists
    assert "trustyai_dashboard.html" in os.listdir()

    # cleanup
    os.remove("trustyai_dashboard.html")
