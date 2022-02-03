# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""Main TrustyAI Python bindings"""
from typing import List
import uuid
import jpype
import jpype.imports
from jpype import _jcustomizer, _jclass

TRUSTY_VERSION = "2.0.0"

#obviously these will need to be changed to the right places
CORE_DEPS = [
    f"/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/explainability-core-2.0.0*",
    "/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/lib/slf4j-api-1.7.30.jar",
    "/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/lib/commons-lang3-3.12.0.jar",
    "/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/lib/commons-math*",
    #"/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/lib/optaplanner-core*",
    "dep/org/optaplanner/optaplanner-core/8.12.0.Final/optaplanner-core-8.12.0.Final.jar",
    "dep/org/kie/kie-api/7.59.0.Final/kie-api-7.59.0.Final.jar",
    "dep/io/micrometer/micrometer-core/1.7.4/micrometer-core-1.7.4.jar",
    "/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/lib/arrow*",
    "/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/lib/flatbuffers*",
    "/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/lib/netty*",
    "/home/rob/Documents/RedHat/kogito/kogito-apps/explainability/explainability-core/target/lib/jackson*",
]

print("== Dev Version ==")
6
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
