# pylint: disable = import-error, import-outside-toplevel, dangerous-default-value, invalid-name, R0801
# pylint: disable = deprecated-module
"""Main TrustyAI Python bindings"""
from distutils.sysconfig import get_python_lib
import glob
import logging
import os
from pathlib import Path
import site
from typing import List
import uuid
import warnings

import jpype
import jpype.imports
from jpype import _jcustomizer, _jclass

DEFAULT_ARGS = (
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    # see https://arrow.apache.org/docs/java/install.html#java-compatibility
    "-Dorg.slf4j.simpleLogger.defaultLogLevel=error",
)


def _get_default_path():
    try:
        default_dep_path = os.path.join(site.getsitepackages()[0], "trustyai", "dep")
    except AttributeError:
        default_dep_path = os.path.join(get_python_lib(), "trustyai", "dep")

    core_deps = [
        f"{default_dep_path}/org/trustyai/explainability-arrow-999-SNAPSHOT.jar",
    ]

    return core_deps, default_dep_path


def init(*args, path=None):
    """init(*args, path=JAVA_DEPENDENCIES)

    Manually initialize the JVM. If you would like to manually specify the Java libraries to be
    imported, for example if you want to use a different version of the Trusty Explainability
    library than is bundled by default, you can do so by calling :func:`init`. If this is not
    manually called, trustyai will use the default set of libraries and automatically initialize
    itself when necessary.

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
        # get default dependencies
        if path is None:
            path, default_dep_path = _get_default_path()
            logging.debug("Checking for dependencies in %s", default_dep_path)

        # check the classpath
        for jar_path in path:
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

        _args = args + DEFAULT_ARGS
        jpype.startJVM(*_args, classpath=path)

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
        print("JVM already initialized")
        warnings.warn("JVM already initialized")

    return True
