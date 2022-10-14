# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value
# pylint: disable = invalid-name, R0801
"""Main TrustyAI Python bindings"""
import os
import logging

# set initialized env variable to 0
import warnings

TRUSTYAI_IS_INITIALIZED = False

if os.getenv("PYTHON_TRUSTY_DEBUG") == "1":
    _LOGGING_LEVEL = logging.DEBUG
else:
    _LOGGING_LEVEL = logging.WARN

logging.basicConfig(level=_LOGGING_LEVEL)


def init():
    """Deprecated manual initializer for the JVM. This function has been replaced by
    automatic initialization when importing the components of the module that require
    JVM access, or by manual user initialization via :func:`trustyai`initializer.init`."""
    warnings.warn(
        "trustyai.init() is now deprecated; the trustyai library will now "
        + "automatically initialize. For manual initialization options, see "
        + "trustyai.initializer.init()"
    )
