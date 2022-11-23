# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
"""Test model provider interface"""

from common import *
from trustyai.model import simple_prediction, counterfactual_prediction,feature, output
from trustyai.utils.data_conversions import numpy_to_prediction_object
import pytest

# test that predictions are created correctly from numpy arrays
def test_predictions_numpy():
    input_values = np.arange(5)
    output_values = np.arange(2)

    pred = simple_prediction(input_values, output_values)
    assert len(pred.getInput().getFeatures()) == 5

    pred = counterfactual_prediction(input_values, output_values)
    assert len(pred.getInput().getFeatures()) == 5


# test that predictions are created correctly from dataframe
def test_predictions_pandas():
    input_values = pd.DataFrame(np.arange(5).reshape(1, -1), columns=list("abcde"))
    output_values = pd.DataFrame(np.arange(2).reshape(1, -1), columns=list("xy"))

    pred = simple_prediction(input_values, output_values)
    assert len(pred.getInput().getFeatures()) == 5
    assert pred.getInput().getFeatures()[0].getName() == "a"

    pred = counterfactual_prediction(input_values, output_values)
    assert pred.getInput().getFeatures()[0].getName() == "a"
    assert len(pred.getInput().getFeatures()) == 5


# test that predictions are created correctly from prediction input + outputs
def test_prediction_pi():
    input_values = numpy_to_prediction_object(np.arange(5).reshape(1, -1), feature)[0]
    output_values = numpy_to_prediction_object(np.arange(2).reshape(1, -1), output)[0]

    pred = simple_prediction(input_values, output_values)
    assert len(pred.getInput().getFeatures()) == 5

    pred = counterfactual_prediction(input_values, output_values)
    assert len(pred.getInput().getFeatures()) == 5


# test that predictions are created correctly from feature+output lists
def test_prediction_featurelist():
    input_values = numpy_to_prediction_object(
        np.arange(5).reshape(1, -1), feature
    )[0].getFeatures()
    output_values = numpy_to_prediction_object(
        np.arange(2).reshape(1, -1), output
    )[0].getOutputs()

    pred = simple_prediction(input_values, output_values)
    assert len(pred.getInput().getFeatures()) == 5

    pred = counterfactual_prediction(input_values, output_values)
    assert len(pred.getInput().getFeatures()) == 5
