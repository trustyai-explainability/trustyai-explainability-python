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
    r"""Create a Java :class:`FeatureDomain`. This represents the valid range of values for a
    particular feature, which is useful when constraining a counterfactual explanation to ensure it
    only recovers valid inputs. For example, if we had a feature that described a person's age, we
    might want to constrain it to the range [0, 125] to ensure the counterfactual explanation
    doesn't return unlikely ages such as -5 or 715.

    Parameters
    ----------
    values : Optional[Union[Tuple, List[str]]]
        The valid values of the feature. If `values` takes the form of:

        * **A tuple of floats or integers:** The feature domain will be a continuous range from
          ``values[0]`` to ``values[1]``.
        * **A list of strings:** The feature domain will be categorical, where `values` contains
          all possible valid feature values.

        Otherwise, the feature domain will be taken as `Empty`, which will mean it will be held
        fixed during the counterfactual explanation.

    Returns
    -------
    :class:`FeatureDomain`
        A Java :class:`FeatureDomain` object, to be used in the :func:`~trustyai.model.feature`
        function.

    """
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
