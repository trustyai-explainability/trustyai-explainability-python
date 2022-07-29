# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""The default initializer"""
import os
from trustyai import initializer

# if trustyai has not yet been initialized, do so now
if os.environ.get("TRUSTYAI_IS_INITIALIZED", "0") == "0":
    initialized = initializer.init()
    os.environ["TRUSTYAI_IS_INITIALIZED"] = "1" if initialized else "0"
