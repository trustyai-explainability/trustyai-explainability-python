# pylint: disable = import-error
"""LIME explanations"""
from org.kie.kogito.explainability.local.lime import (
    LimeConfig as _LimeConfig,
    LimeExplainer as _LimeExplainer,
)

LimeConfig = _LimeConfig
LimeExplainer = _LimeExplainer
