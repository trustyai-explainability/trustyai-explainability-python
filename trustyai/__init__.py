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

        @_jcustomizer.JConversion("java.util.List", exact=List)
        def _JListConvert(_, py_list: List):
            return _jclass.JClass("java.util.Arrays").asList(py_list)

        @_jcustomizer.JConversion("java.util.UUID", instanceof=uuid.UUID)
        def _JUUIDConvert(_, obj):
            return UUID.fromString(str(obj))

    except OSError:
        print("JVM already started")
