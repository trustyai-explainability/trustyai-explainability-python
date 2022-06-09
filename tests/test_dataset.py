# pylint: disable=import-error, wrong-import-position, wrong-import-order, R0801
"""Test suite for the Dataset structure"""

from common import *

from java.util import Random
from pytest import approx
import pandas as pd
import numpy as np
import uuid

from trustyai.model import Dataset, Type


jrandom = Random()
jrandom.setSeed(0)

def generate_test_df():
    data = {
            'x1': np.random.uniform(low=100, high=200, size=100),
            'x2': np.random.uniform(low=5000, high=10000, size=100),
            'x3': [str(uuid.uuid4()) for _ in range(100)],
            'x4': np.random.randint(low=0, high=42, size=100),
            'select': np.random.choice(a=[False, True], size=100)
    }
    return pd.DataFrame(data=data)


def test_no_output():
    """Checks whether we have an output when specifying none"""
    df = generate_test_df()
    dataset = Dataset.from_df(df)
    outputs = dataset.outputs[0].outputs
    assert len(outputs) == 1
    assert outputs[0].name == 'select'

def test_outputs():
    """Checks whether we have the correct specified outputs"""
    df = generate_test_df()
    dataset = Dataset.from_df(df, outputs=["x2", "x3"])
    outputs = dataset.outputs[0].outputs
    assert len(outputs) == 2
    assert outputs[0].name == 'x2' and outputs[1].name == 'x3'

def test_shape():
    """Checks whether we have the correct shape"""
    df = generate_test_df()
    dataset = Dataset.from_df(df, outputs=["x4"])
    assert len(dataset.outputs) == 100
    assert len(dataset.inputs) == 100
    assert len(dataset.data) == 100

    assert len(dataset.inputs[0].features) == 4
    assert len(dataset.outputs[0].outputs) == 1

def test_types():
    """Checks whether we have the correct shape"""
    df = generate_test_df()
    dataset = Dataset.from_df(df, outputs=["x4"])
    features = dataset.inputs[0].features
    assert features[0].type == Type.NUMBER and features[0].name == 'x1'
    assert features[1].type == Type.NUMBER and features[1].name == 'x2'
    assert features[2].type == Type.CATEGORICAL and features[2].name == 'x3'
    assert features[3].type == Type.BOOLEAN and features[3].name == 'select'
    outputs = dataset.outputs[0].outputs
    assert outputs[0].type == Type.NUMBER and outputs[0].name == 'x4'
