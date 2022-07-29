import os

import pytest
from multiprocessing import Process, Value
import sys


# slightly hacky functions to make sure the test process does not see the trustyai initialization
# from commons.py
def manual_initializer_process(initial_state, final_state):
    from trustyai import initializer
    initial_state.value = int(os.environ.get("TRUSTYAI_IS_INITIALIZED", "0"))
    initializer.init(path=initializer._get_default_path()[0])

    # test imports work
    from trustyai.explainers import SHAPExplainer

    # test initialization is set
    final_state.value = int(os.environ["TRUSTYAI_IS_INITIALIZED"])


def default_initializer_process_mod(initial_state, final_state):
    initial_state.value = int(os.environ.get("TRUSTYAI_IS_INITIALIZED", "0"))
    import trustyai.model

    # test initialization is set
    final_state.value = int(os.environ["TRUSTYAI_IS_INITIALIZED"])


def default_initializer_process_exp(initial_state, final_state):
    initial_state.value = int(os.environ.get("TRUSTYAI_IS_INITIALIZED", "0"))
    import trustyai.explainers

    # test initialization is set
    final_state.value = int(os.environ["TRUSTYAI_IS_INITIALIZED"])


# test that import ta.explainers correctly initializes
def test_default_initialization_explainers():
    os.environ["TRUSTYAI_IS_INITIALIZED"] = "0"
    initial_state = Value('i', -1)
    final_state = Value('i', -1)
    process = Process(target=default_initializer_process_exp, args=(initial_state, final_state))
    process.start()
    process.join()
    assert initial_state.value == 0
    assert final_state.value == 1


# test that import ta.models correctly initializes
def test_default_initialization_models():
    os.environ["TRUSTYAI_IS_INITIALIZED"] = "0"
    initial_state = Value('i', -1)
    final_state = Value('i', -1)
    process = Process(target=default_initializer_process_mod, args=(initial_state, final_state))
    process.start()
    process.join()
    assert initial_state.value == 0
    assert final_state.value == 1


# test that manually initializing also works
def test_manual_initialization():
    os.environ["TRUSTYAI_IS_INITIALIZED"] = "0"
    initial_state = Value('i', -1)
    final_state = Value('i', -1)
    process = Process(target=manual_initializer_process, args=(initial_state, final_state))
    process.start()
    process.join()
    assert initial_state.value == 0
    assert final_state.value == 1


if __name__ == "__main__":
    test_manual_initialization()
