import multiprocessing
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

functions = [
    manual_initializer_process,
    default_initializer_process_exp,
    default_initializer_process_mod
]

# test that manually initializing also works
@pytest.mark.parametrize("function",functions)
def test_initialization(function):
    ctx = multiprocessing.get_context("spawn")
    initial_state = Value('i', -1)
    final_state = Value('i', -1)
    process = ctx.Process(target=function, args=(initial_state, final_state))
    process.start()
    process.join()
    assert initial_state.value == 0
    assert final_state.value == 1
