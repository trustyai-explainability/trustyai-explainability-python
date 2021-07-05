# pylint: disable = import-error
"""Counterfactual explanations"""
from org.kie.kogito.explainability.local.counterfactual import (
    CounterfactualExplainer as _CounterfactualExplainer,
    CounterfactualConfigurationFactory as _CounterfactualConfigurationFactory,
)

CounterfactualExplainer = _CounterfactualExplainer
CounterfactualConfigurationFactory = _CounterfactualConfigurationFactory
