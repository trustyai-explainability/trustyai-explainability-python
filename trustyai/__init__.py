# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""Main TrustyAI Python bindings"""
import os
import site
import uuid
from pathlib import Path
from typing import List
import glob
import logging

import jpype
import jpype.imports
from jpype import _jcustomizer, _jclass

TRUSTY_VERSION = "1.20.0.Final"
DEFAULT_DEP_PATH = os.path.join(site.getsitepackages()[0], "trustyai", "dep")

CORE_DEPS = [
    f"{DEFAULT_DEP_PATH}/org/kie/kogito/explainability-core/{TRUSTY_VERSION}/*",
    f"{DEFAULT_DEP_PATH}/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar",
    f"{DEFAULT_DEP_PATH}/org/apache/commons/commons-lang3/3.12.0/commons-lang3-3.12.0.jar",
    f"{DEFAULT_DEP_PATH}/org/optaplanner/optaplanner-core-impl/8.20.0.Final/"
    f"optaplanner-core-impl-8.20.0.Final.jar",
    f"{DEFAULT_DEP_PATH}/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar",
    f"{DEFAULT_DEP_PATH}/org/kie/kie-api/8.20.0.Beta/kie-api-8.20.0.Beta.jar",
    f"{DEFAULT_DEP_PATH}/io/micrometer/micrometer-core/1.8.3/micrometer-core-1.8.3.jar",
]

ARROW_DEPS = [
    f"{DEFAULT_DEP_PATH}/io/netty/netty-buffer/4.1.68.Final/netty-buffer-4.1.68.Final.jar",
    f"{DEFAULT_DEP_PATH}/io/netty/netty-common/4.1.68.Final/netty-common-4.1.68.Final.jar",
    f"{DEFAULT_DEP_PATH}/org/apache/arrow/arrow-format/7.0.0/arrow-format-7.0.0.jar",
    f"{DEFAULT_DEP_PATH}/org/apache/arrow/arrow-memory-core/7.0.0/arrow-memory-core-7.0.0.jar",
    f"{DEFAULT_DEP_PATH}/org/apache/arrow/arrow-memory-netty/7.0.0/arrow-memory-netty-7.0.0.jar",
    f"{DEFAULT_DEP_PATH}/org/apache/arrow/arrow-vector/7.0.0/arrow-vector-7.0.0.jar",
    f"{DEFAULT_DEP_PATH}/org/trustyai/arrow-converters-0.0.1.jar",
    f"{DEFAULT_DEP_PATH}/com/google/flatbuffers/flatbuffers-java/1.12.0/"
    f"flatbuffers-java-1.12.0.jar",
    f"{DEFAULT_DEP_PATH}/com/fasterxml/jackson/core/jackson-core/2.13.1/jackson-core-2.13.1.jar",
    f"{DEFAULT_DEP_PATH}/com/fasterxml/jackson/core/jackson-databind/2.13.1/"
    f"jackson-databind-2.13.1.jar",
    f"{DEFAULT_DEP_PATH}/com/fasterxml/jackson/core/jackson-annotations/2.13.1/"
    f"jackson-annotations-2.13.1.jar",
]

CORE_DEPS += ARROW_DEPS

if os.getenv("PYTHON_TRUSTY_DEBUG") == "1":
    _logging_level = logging.DEBUG
else:
    _logging_level = logging.WARN

logging.basicConfig(level=_logging_level)


def init(*args, path=CORE_DEPS):
    """Initialise Java binding"""
    # Launch the JVM
    try:
        # check the classpath
        logging.debug("Checking for dependencies in %s", DEFAULT_DEP_PATH)
        for jar_path in CORE_DEPS:
            if "*" not in jar_path:
                jar_path_exists = Path(jar_path).exists()
            else:
                jar_path_exists = any(
                    Path(fp).exists() for fp in glob.glob(jar_path) if ".jar" in fp
                )
            if jar_path_exists:
                logging.debug("JAR %s found.", jar_path)
            else:
                logging.error("JAR %s not found.", jar_path)

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
