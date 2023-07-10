"""
Wrapper module for TSICEExplainer from aix360.
Original at https://github.com/Trusted-AI/AIX360/
"""
# pylint: disable=too-many-arguments
from typing import Callable, List, Optional, Union

from aix360.algorithms.tsice import TSICEExplainer as TSICEExplainerAIX
from aix360.algorithms.tsutils.tsperturbers import TSPerturber


class TSICEExplainer(TSICEExplainerAIX):
    """
    Wrapper for TSICEExplainer from aix360.
    """

    def __init__(
        self,
        forecaster: Callable,
        input_length: int,
        forecast_lookahead: int,
        n_variables: int = 1,
        n_exogs: int = 0,
        n_perturbations: int = 25,
        features_to_analyze: Optional[List[str]] = None,
        perturbers: Optional[List[Union[TSPerturber, dict]]] = None,
        explanation_window_start: Optional[int] = None,
        explanation_window_length: int = 10,
    ):
        super().__init__(
            forecaster=forecaster,
            input_length=input_length,
            forecast_lookahead=forecast_lookahead,
            n_variables=n_variables,
            n_exogs=n_exogs,
            n_perturbations=n_perturbations,
            features_to_analyze=features_to_analyze,
            perturbers=perturbers,
            explanation_window_start=explanation_window_start,
            explanation_window_length=explanation_window_length,
        )

    def explain(self, inputs, outputs=None, **kwargs):
        """
        Explain the model's prediction on X.
        """
        return super().explain_instance(inputs, y=outputs, **kwargs)
