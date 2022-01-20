# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""Main TrustyAI Python bindings"""
from typing import List
import uuid
import jpype
import jpype.imports
from jpype import _jcustomizer, _jclass

TRUSTY_VERSION = "1.12.0.Final"
CORE_DEPS = [
    f"./dep/org/kie/kogito/explainability-core/{TRUSTY_VERSION}/*",
    "./dep/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar",
    "./dep/org/apache/commons/commons-lang3/3.12.0/commons-lang3-3.12.0.jar",
]


def init(*args, path=CORE_DEPS):
    """Initialise Java binding"""
    # Launch the JVM
    try:
        jpype.startJVM(*args, classpath=path)

        from java.lang import Thread

        if not Thread.isAttached:
            jpype.attachThreadToJVM()

        from java.util import UUID
        from org.kie.kogito.explainability.local.counterfactual import CounterfactualResult
        from org.kie.kogito.explainability.local.counterfactual.entities import CounterfactualEntity
        from org.kie.kogito.explainability.model import Feature

        @_jcustomizer.JConversion("java.util.List", exact=List)
        def _JListConvert(_, py_list: List):
            return _jclass.JClass("java.util.Arrays").asList(py_list)

        @_jcustomizer.JConversion("java.util.UUID", instanceof=uuid.UUID)
        def _JUUIDConvert(_, obj):
            return UUID.fromString(str(obj))

        # implicit conversion
        @_jcustomizer.JImplementationFor("org.kie.kogito.explainability.local.counterfactual.CounterfactualResult")
        # pylint: disable=no-member
        class _JCounterfactualResult:
            """Java CounterfactualResult implicit methods"""

            @property
            def entities(self) -> List[CounterfactualEntity]:
                """Return entities"""
                return self.getEntities()

            @property
            def output(self):
                """Return PredictionOutput"""
                return self.getOutput()

        @_jcustomizer.JImplementationFor(
            "org.kie.kogito.explainability.local.counterfactual.entities.CounterfactualEntity")
        # pylint: disable=no-member
        class _JCounterfactualEntity:
            """Java DoubleEntity implicit methods"""

            def as_feature(self) -> Feature:
                """Return as feature"""
                return self.asFeature()

        @_jcustomizer.JImplementationFor("org.kie.kogito.explainability.model.Feature")
        # pylint: disable=no-member
        class _JFeature:
            """Java Feature implicit methods"""

            @property
            def name(self):
                """Return name"""
                return self.getName()

            @property
            def type(self):
                """Return type"""
                return self.getType()

            @property
            def value(self):
                """Return value"""
                return self.getValue()

            def __str__(self):
                return self.toString()

        @_jcustomizer.JImplementationFor("org.kie.kogito.explainability.model.Value")
        # pylint: disable=no-member
        class _JValue:
            """Java Value implicit methods"""

            def as_string(self) -> str:
                """Return as string"""
                return self.asString()

            def as_number(self) -> float:
                """Return as number"""
                return self.asNumber()

            def __str__(self):
                return self.toString()

    except OSError:
        print("JVM already started")
