# pylint: disable = import-error, import-outside-toplevel
"""Main TrustyAI Python bindings"""
import sys
import jpype
import jpype.imports


def init(
    path="../kogito-apps/explainability/explainability-core/target/explainability-core-2.0.0-SNAPSHOT.jar",  # pylint: disable = line-too-long
):
    """Initialise Java binding"""
    # Launch the JVM
    jpype.startJVM(classpath=[path])

    from org.kie.kogito.explainability.utils import DataUtils

    setattr(sys.modules[__name__], "DataUtils", DataUtils)
