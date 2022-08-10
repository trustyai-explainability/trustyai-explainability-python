# pylint: disable = import-error, invalid-name, wrong-import-order
"""General model classes"""
from trustyai import _default_initializer
from org.kie.trustyai.explainability.utils import (
    ExplainabilityMetrics as _ExplainabilityMetrics,
)

ExplainabilityMetrics = _ExplainabilityMetrics
