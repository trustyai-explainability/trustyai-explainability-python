# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""Main TrustyAI Python bindings"""
import os
import site
import uuid
from typing import List

import jpype
import jpype.imports
from jpype import _jcustomizer, _jclass

TRUSTY_VERSION = "1.18.0.Final"
DEFAULT_DEP_PATH = os.path.join(site.getsitepackages()[0], "trustyai", "dep")

CORE_DEPS = [
    f"{DEFAULT_DEP_PATH}/org/kie/kogito/explainability-core/{TRUSTY_VERSION}/*",
    f"{DEFAULT_DEP_PATH}/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar",
    f"{DEFAULT_DEP_PATH}/org/apache/commons/commons-lang3/3.12.0/commons-lang3-3.12.0.jar",
    f"{DEFAULT_DEP_PATH}/org/optaplanner/optaplanner-core-impl/8.18.0.Final/"
    f"optaplanner-core-impl-8.18.0.Final.jar",
    f"{DEFAULT_DEP_PATH}/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar",
    f"{DEFAULT_DEP_PATH}/org/kie/kie-api/8.18.0.Beta/kie-api-8.18.0.Beta.jar",
    f"{DEFAULT_DEP_PATH}/io/micrometer/micrometer-core/1.8.2/micrometer-core-1.8.2.jar",
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
