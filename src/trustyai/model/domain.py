# pylint: disable = import-error
"""Conversion method between Python and TrustyAI Java types"""
from typing import Optional, Tuple, List, Union

from jpype import _jclass

from org.kie.trustyai.explainability.model.domain import (
    FeatureDomain,
    NumericalFeatureDomain,
    CategoricalFeatureDomain,
    CategoricalNumericalFeatureDomain,
    ObjectFeatureDomain,
    EmptyFeatureDomain,
)


def feature_domain(values: Optional[Union[Tuple, List]]) -> Optional[FeatureDomain]:
    r"""Create a Java :class:`FeatureDomain`. This represents the valid range of values for a
    particular feature, which is useful when constraining a counterfactual explanation to ensure it
    only recovers valid inputs. For example, if we had a feature that described a person's age, we
    might want to constrain it to the range [0, 125] to ensure the counterfactual explanation
    doesn't return unlikely ages such as -5 or 715.

    Parameters
    ----------
    values : Optional[Union[Tuple, List]]
        The valid values of the feature. If ``values`` takes the form of:

        * **A tuple of floats or integers**: The feature domain will be a continuous range from
          ``values[0]`` to ``values[1]``.
        * **A list of floats or integers**: The feature domain will be a *numeric* categorical,
          where `values` contains all possible valid feature values.
        * **A list of strings**: The feature domain will be a *string* categorical, where ``values``
          contains all possible valid feature values.
        * **A list of objects**: The feature domain will be an *object* categorical, where
          ``values`` contains all possible valid feature values. These may present an issue if the
          objects are not natively Java serializable.

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
        if isinstance(values, tuple):
            assert isinstance(values[0], (float, int)) and isinstance(
                values[1], (float, int)
            )
            assert len(values) == 2, (
                "Tuples passed as domain values must only contain"
                " two values that define the (minimum, maximum) of the domain"
            )
            domain = NumericalFeatureDomain.create(values[0], values[1])

        elif isinstance(values, list):
            java_array = _jclass.JClass("java.util.Arrays").asList(values)
            if isinstance(values[0], bool) and isinstance(values[1], bool):
                domain = ObjectFeatureDomain.create(java_array)
            elif isinstance(values[0], (float, int)) and isinstance(
                values[1], (float, int)
            ):
                domain = CategoricalNumericalFeatureDomain.create(java_array)
            elif isinstance(values[0], str):
                domain = CategoricalFeatureDomain.create(java_array)
            else:
                domain = ObjectFeatureDomain.create(java_array)

        else:
            domain = EmptyFeatureDomain.create()
    return domain
