# pylint: disable = import-error, invalid-name, wrong-import-order
"""General model classes"""

from jpype._jproxy import _createJProxy, _createJProxyDeferred
from trustyai import _default_initializer

from org.kie.kogito.explainability import (
    TestUtils as _TestUtils,
    Config as _Config,
)

TestUtils = _TestUtils
Config = _Config


def JImplementsWithDocstring(*interfaces, deferred=False, **kwargs):
    """JPype's JImplements decorator overwrites the docstring of any annotated functions. This
    is a quick hack to preserve docstrings across the jproxy process."""
    if deferred:

        def JProxyCreator(cls):
            proxy_class = _createJProxyDeferred(cls, *interfaces, **kwargs)
            proxy_class.__doc__ = cls.__doc__
            proxy_class.__name__ = cls.__name__
            return proxy_class

    else:

        def JProxyCreator(cls):
            proxy_class = _createJProxy(cls, *interfaces, **kwargs)
            proxy_class.__doc__ = cls.__doc__
            proxy_class.__name__ = cls.__name__
            return proxy_class

    return JProxyCreator
