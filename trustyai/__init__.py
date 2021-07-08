# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""Main TrustyAI Python bindings"""
from typing import List
import uuid
import jpype
import jpype.imports
from jpype import _jcustomizer


def init(
    *args,
    path=[
        "./dep/org/kie/kogito/explainability-core/1.5.0.Final/*",
        "./dep/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar",
        "./dep/org/apache/commons/commons-lang3/3.8.1/commons-lang3-3.8.1.jar",
    ],  # pylint: disable = line-too-long
):
    """Initialise Java binding"""
    # Launch the JVM
    try:
        jpype.startJVM(*args, classpath=path)

        from java.lang import Thread

        if not Thread.isAttached:
            jpype.attachThreadToJVM()

        from trustyai.utils import toJList
        from java.util import UUID

        @_jcustomizer.JConversion("java.util.List", exact=List)
        def _JSequenceConvert(_, obj):
            return toJList(obj)

        @_jcustomizer.JConversion("java.util.UUID", instanceof=uuid.UUID)
        def _JSequenceConvert(_, obj):
            return UUID.fromString(str(obj))

    except OSError:
        print("JVM already started")
