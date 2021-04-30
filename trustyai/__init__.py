# pylint: disable = import-error, import-outside-toplevel
"""Main TrustyAI Python bindings"""
import sys
import jpype
import jpype.imports


def init(
    path="./dep/org/kie/kogito/explainability-core/1.5.0.Final/*",  # pylint: disable = line-too-long
):
    """Initialise Java binding"""
    # Launch the JVM
    jpype.startJVM(classpath=[path])
