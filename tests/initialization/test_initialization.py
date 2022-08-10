import multiprocessing
import os

import pytest
from multiprocessing import Process, Value
import sys


# slightly hacky functions to make sure the test process does not see the trustyai initialization
# from commons.py
def test_manual_initializer_process():
    import trustyai
    from trustyai import initializer
    initial_state = trustyai.TRUSTYAI_IS_INITIALIZED
    initializer.init(path=initializer._get_default_path()[0])

    # test imports work
    from trustyai.explainers import SHAPExplainer

    # test initialization is set
    final_state = trustyai.TRUSTYAI_IS_INITIALIZED
    assert initial_state == False
    assert final_state == True


def test_default_initializer_process_mod():
    import trustyai
    initial_state = trustyai.TRUSTYAI_IS_INITIALIZED
    import trustyai.model

    # test initialization is set
    final_state = trustyai.TRUSTYAI_IS_INITIALIZED
    assert initial_state == False
    assert final_state == True


def test_default_initializer_process_exp():
    import trustyai
    initial_state = trustyai.TRUSTYAI_IS_INITIALIZED
    import trustyai.explainers

    # test initialization is set
    final_state = trustyai.TRUSTYAI_IS_INITIALIZED
    assert initial_state == False
    assert final_state == True
