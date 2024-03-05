"""Test suite for TrustyAI metrics service data conversions"""
import json
import os
import random
import unittest
import numpy as np
import pandas as pd

from trustyai.utils.extras.metrics_service import (
    json_to_df,
    df_to_json
)

def generate_json_data(batch_list, data_path):
    for batch in batch_list:
        data = {
        "inputs": [
            {"name": "test_data_input",
                "shape": [1, 100],
                "datatype": "FP64",
                "data": [random.uniform(a=100, b=200) for i in range(100)]
                }
            ]
        }
        for batch in batch_list:
            with open(data_path + f"{batch}.json", 'w', encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)


def generate_test_df():
    data = {
        '0': np.random.uniform(low=100, high=200, size=100),
        '1': np.random.uniform(low=5000, high=10000, size=100),
        '2': np.random.uniform(low=100, high=200, size=100),
        '3': np.random.uniform(low=5000, high=10000, size=100),
        '4': np.random.uniform(low=5000, high=10000, size=100)
    }
    return pd.DataFrame(data=data)


class TestMetricsService(unittest.TestCase):
    def setUp(self):
        self.df = generate_test_df()
        self.data_path = "data/"
        if not os.path.exists(self.data_path):
            os.mkdir("data/")
        self.batch_list = list(range(0, 5))

    def test_json_to_df(self):
        """Test json data to pandas dataframe conversion"""
        generate_json_data(batch_list=self.batch_list, data_path=self.data_path)
        df = json_to_df(self.data_path, self.batch_list)
        n_rows, n_cols = 0, 0
        for batch in self.batch_list:
            file = self.data_path + f"{batch}.json"
            with open(file, encoding="utf8") as f:
                data = json.load(f)["inputs"][0]
                n_rows += data["shape"][0]
                n_cols = data["shape"][1]
        self.assertEqual(df.shape, (n_rows, n_cols))


    def test_df_to_json(self):
        """Test pandas dataframe to json data conversion"""
        df = generate_test_df()
        name = 'test_data_input'
        json_file = 'data/test.json'
        df_to_json(df, name, json_file)
        with open(json_file, encoding="utf8") as f:
            data = json.load(f)["inputs"][0]
        n_rows = data["shape"][0]
        n_cols =  data["shape"][1]
        self.assertEqual(df.shape, (n_rows, n_cols))

if __name__ == "__main__":
    unittest.main()
