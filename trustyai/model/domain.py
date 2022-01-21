# pylint: disable = import-error
"""Conversion method between Python and TrustyAI Java types"""
from typing import Optional, Tuple, List, Union

from jpype import _jclass
from org.kie.kogito.explainability.model.domain import (
    FeatureDomain,
    NumericalFeatureDomain,
    CategoricalFeatureDomain,
    EmptyFeatureDomain,
)


def feature_domain(
    values: Optional[Union[Tuple, List[str]]]
) -> Optional[FeatureDomain]:
    """Create Java FeatureDomain from a Python tuple or list"""
    if not values:
        domain = EmptyFeatureDomain.create()
    else:
        if isinstance(values[0], (float, int)):
            domain = NumericalFeatureDomain.create(values[0], values[1])
        elif isinstance(values[0], str):
            domain = CategoricalFeatureDomain.create(
                _jclass.JClass("java.util.Arrays").asList(values)
            )
        else:
            domain = EmptyFeatureDomain.create()
    return domain
