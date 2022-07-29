# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""Main TrustyAI Python bindings"""
import os
import logging

# set initialized env variable to 0
os.environ['TRUSTYAI_IS_INITIALIZED'] = "0"

if os.getenv("PYTHON_TRUSTY_DEBUG") == "1":
    _logging_level = logging.DEBUG
else:
    _logging_level = logging.WARN

logging.basicConfig(level=_logging_level)
