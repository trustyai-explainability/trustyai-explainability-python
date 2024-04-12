"""Python client for TrustyAI metrics"""

from typing import List
import json
import datetime as dt
import pandas as pd
import requests
import matplotlib.pyplot as plt

from trustyai.utils.api.api import TrustyAIApi


def json_to_df(data_path: str, batch_list: List[int]) -> pd.DataFrame:
    """
    Converts batched data in json files to a single pandas DataFrame
    """
    final_df = pd.DataFrame()
    for batch in batch_list:
        file = data_path + f"{batch}.json"
        with open(file, encoding="utf8") as train_file:
            batch_data = json.load(train_file)["inputs"][0]
            batch_df = pd.DataFrame.from_dict(batch_data["data"]).T
            final_df = pd.concat([final_df, batch_df])
    return final_df


def df_to_json(final_df: pd.DataFrame, name: str, json_file: str) -> None:
    """
    Converts pandas DataFrame to json file
    """
    inputs = [
        {
            "name": name,
            "shape": list(final_df.shape),
            "datatype": "FP64",
            "data": final_df.values.tolist(),
        }
    ]
    data_dict = {"inputs": inputs}
    with open(json_file, "w", encoding="utf8") as outfile:
        json.dump(data_dict, outfile)


class TrustyAIMetricsService:
    """
    Executes and returns queries from TrustyAI service on ODH
    """

    def __init__(self, token: str, namespace: str, verify=True):
        """
        :param token: OpenShift login token
        :param namespace: model namespace
        :param verify: enable SSL verification for requests
        """
        self.token = token
        self.namespace = namespace
        self.trusty_url = TrustyAIApi().get_service_route(
            name="trustyai-service", namespace=self.namespace
        )
        self.thanos_url = TrustyAIApi().get_service_route(
            name="thanos-querier", namespace="openshift-monitoring"
        )
        self.headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }
        self.verify = verify

    def upload_payload_data(self, json_file: str, timeout=5) -> None:
        """
        Uploads data to TrustyAI service
        """
        with open(json_file, "r", encoding="utf8") as file:
            response = requests.post(
                f"{self.trusty_url}/data/upload",
                data=file,
                headers=self.headers,
                verify=self.verify,
                timeout=timeout,
            )
        if response.status_code == 200:
            print("Data sucessfully uploaded to TrustyAI service")
        else:
            print(f"Error {response.status_code}: {response.reason}")

    def get_model_metadata(self, timeout=5):
        """
        Retrieves model data from TrustyAI
        """
        response = requests.get(
            f"{self.trusty_url}/info",
            headers=self.headers,
            verify=self.verify,
            timeout=timeout,
        )
        if response.status_code == 200:
            model_metadata = json.loads(response.text)
            return model_metadata
        raise RuntimeError(f"Error {response.status_code}: {response.reason}")

    def label_data_fields(self, payload: str, timeout=5):
        """
        Assigns feature names to model input data
        """

        def print_name_mapping(self):
            response = requests.get(
                f"{self.trusty_url}/info",
                headers=self.headers,
                verify=self.verify,
                timeout=timeout,
            )
            name_mapping = json.loads(response.text)[0]
            for key, val in name_mapping["data"]["inputSchema"]["nameMapping"].items():
                print(f"{key} -> {val}")

        response = requests.get(
            f"{self.trusty_url}/info",
            headers=self.headers,
            verify=self.verify,
            timeout=timeout,
        )
        input_data_fields = list(
            json.loads(response.text)[0]["data"]["inputSchema"]["items"].keys()
        )
        input_mapping_keys = list(payload["inputMapping"].keys())
        if len(list(set(input_mapping_keys) - set(input_data_fields))) == 0:
            response = requests.post(
                f"{self.trusty_url}/info/names",
                json=payload,
                headers=self.headers,
                verify=self.verify,
                timeout=timeout,
            )
            if response.status_code == 200:
                print_name_mapping(self)
                return response.text
            print(f"Error {response.status_code}: {response.reason}")
        raise ValueError("Field does not exist")

    def get_metric_request(
        self, payload: str, metric: str, reoccuring: bool, timeout=5
    ):
        """
        Retrieve or schedule a metric request
        """
        if reoccuring:
            response = requests.post(
                f"{self.trusty_url}/metrics/{metric}/request",
                json=payload,
                headers=self.headers,
                verify=self.verify,
                timeout=timeout,
            )
        else:
            response = requests.post(
                f"{self.trusty_url}/metrics/{metric}",
                json=payload,
                headers=self.headers,
                verify=self.verify,
                timeout=timeout,
            )
        if response.status_code == 200:
            return response.text
        raise RuntimeError(f"Error {response.status_code}: {response.reason}")

    def upload_data_to_model(self, model_name: str, json_file: str, timeout=5):
        """
        Sends an inference request to the model
        """
        model_route = TrustyAIApi().get_service_route(
            name=model_name, namespace=self.namespace
        )
        with open(json_file, encoding="utf8") as batch_file:
            response = requests.post(
                url=f"https://{model_route}/infer",
                data=batch_file,
                headers=self.headers,
                verify=self.verify,
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.text
            raise RuntimeError(f"Error {response.status_code}: {response.reason}")

    def get_metric_data(self, metric: str, time_interval: List[str], timeout=5):
        """
        Retrives metric data for a specific range in time for each subcategory in data field
        """
        metric_df = pd.DataFrame()
        for subcategory in list(
            self.get_model_metadata()[0]["data"]["inputSchema"]["nameMapping"].values()
        ):
            params = {
                "query": f"{metric}{{subcategory='{subcategory}'}}{time_interval}"
            }

            response = requests.get(
                f"{self.thanos_url}/api/v1/query?",
                params=params,
                headers=self.headers,
                verify=self.verify,
                timeout=timeout,
            )
            if response.status_code == 200:
                if "timestamp" in metric_df.columns:
                    pass
                else:
                    metric_df["timestamp"] = [
                        item[0]
                        for item in json.loads(response.text)["data"]["result"][0][
                            "values"
                        ]
                    ]
                metric_df[subcategory] = [
                    item[1]
                    for item in json.loads(response.text)["data"]["result"][0]["values"]
                ]
            else:
                raise RuntimeError(f"Error {response.status_code}: {response.reason}")

        metric_df["timestamp"] = metric_df["timestamp"].apply(
            lambda epoch: dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
        )
        return metric_df

    @staticmethod
    def plot_metric(metric_df: pd.DataFrame, metric: str):
        """
        Plots a line for each subcategory in the pandas DataFrame returned by get_metric_request
        with the timestamp on x-axis and specified metric on the y-axis
        """
        plt.figure(figsize=(12, 5))
        for col in metric_df.columns[1:]:
            plt.plot(metric_df["timestamp"], metric_df[col])
        plt.xlabel("timestamp")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend(metric_df.columns[1:])
        plt.tight_layout()
        plt.show()
