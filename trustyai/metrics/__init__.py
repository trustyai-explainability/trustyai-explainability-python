# pylint: disable = import-error, invalid-name, wrong-import-order, no-name-in-module
"""General model classes"""
from trustyai import _default_initializer  # pylint: disable=unused-import
from org.kie.kogito.explainability.utils import (
    ExplainabilityMetrics as _ExplainabilityMetrics,
)

ExplainabilityMetrics = _ExplainabilityMetrics
