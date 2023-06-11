from typing import Callable, List, Union

from aix360.algorithms.tsice import TSICEExplainer as TSICEExplainerAIX
from aix360.algorithms.tsutils.tsperturbers import TSPerturber


class TSICEExplainer(TSICEExplainerAIX):
    def __init__(self,
                 forecaster: Callable,
                 input_length: int,
                 forecast_lookahead: int,
                 n_variables: int = 1,
                 n_exogs: int = 0,
                 n_perturbations: int = 25,
                 features_to_analyze: List[str] = None,
                 perturbers: List[Union[TSPerturber, dict]] = None,
                 explanation_window_start: int = None,
                 explanation_window_length: int = 10,
                 ):
        super().__init__(forecaster=forecaster, input_length=input_length, forecast_lookahead=forecast_lookahead,
                         n_variables=n_variables, n_exogs=n_exogs, n_perturbations=n_perturbations,
                         features_to_analyze=features_to_analyze, perturbers=perturbers,
                         explanation_window_start=explanation_window_start,
                         explanation_window_length=explanation_window_length)

    def explain(self, X, y=None, **kwargs):
        return super().explain(X, y=y, **kwargs)
