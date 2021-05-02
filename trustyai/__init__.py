# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value
"""Main TrustyAI Python bindings"""
import sys
import jpype
import jpype.imports


def init(
    path=[
        "./dep/org/kie/kogito/explainability-core/1.5.0.Final/*",
        "./dep/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar",
        "./dep/org/apache/commons/commons-lang3/3.8.1/commons-lang3-3.8.1.jar",
    ],  # pylint: disable = line-too-long
):
    """Initialise Java binding"""
    # Launch the JVM
    try:
        jpype.startJVM(classpath=path)

        from java.lang import Thread

        if not Thread.isAttached:
            jpype.attachThreadToJVM()
    except OSError:
        print("JVM already started")
