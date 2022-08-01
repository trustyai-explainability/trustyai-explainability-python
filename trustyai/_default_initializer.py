# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""The default initializer"""
import trustyai
from trustyai import initializer

if not trustyai.TRUSTYAI_IS_INITIALIZED:
    trustyai.TRUSTYAI_IS_INITIALIZED = initializer.init()