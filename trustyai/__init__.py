# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
"""Main TrustyAI Python bindings"""
import os
import site
import uuid
from pathlib import Path
from typing import List
import glob
import logging
from distutils.sysconfig import get_python_lib

import jpype
import jpype.imports
from jpype import _jcustomizer, _jclass

TRUSTY_VERSION = "1.22.1.Final"

try:
    DEFAULT_DEP_PATH = os.path.join(site.getsitepackages()[0], "trustyai", "dep")
except AttributeError:
    DEFAULT_DEP_PATH = os.path.join(get_python_lib(), "trustyai", "dep")

CORE_DEPS = [
    f"{DEFAULT_DEP_PATH}/org/trustyai/explainability-core-2.0.0-SNAPSHOT.jar",
    f"{DEFAULT_DEP_PATH}/org/trustyai/explainability-core-2.0.0-SNAPSHOT-tests.jar",
]

ARROW_DEPS = [
    f"{DEFAULT_DEP_PATH}/org/trustyai/arrow-converters-0.0.1.jar",
]

CORE_DEPS += ARROW_DEPS

if os.getenv("PYTHON_TRUSTY_DEBUG") == "1":
    _logging_level = logging.DEBUG
else:
    _logging_level = logging.WARN

logging.basicConfig(level=_logging_level)


def init(*args, path=CORE_DEPS):
    """init(*args, path=JAVA_DEPENDENCIES)

    Initialize the Java bindings. We recommend calling this directly after importing trustyai,
    to avoid import errors:

    ::

        import trustyai
        trustyai.init()


    Parameters
    ----------
    args: list
        List of args to be passed to ``jpype.startJVM``. See the
        `JPype manual <https://jpype.readthedocs.io/en/latest/api.html#jpype.startJVM>`_
        for more details.
    path: list[str]
        List of jar files to add the Java class path. By default, this will add the necessary
        dependencies of the TrustyAI Java library.

    """
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
