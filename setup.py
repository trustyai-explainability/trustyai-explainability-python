import os
from setuptools import setup
from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
import site

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

TRUSTY_VERSION = "1.22.1.Final"


class PostInstall(install):
    """Fetch TrustyAI explainability JARs from Maven Central"""

    def run(self):
        install.run(self)
        try:
            _ROOT = os.path.join(site.getsitepackages()[0], "trustyai", "dep")
        except AttributeError:
            _ROOT = os.path.join(get_python_lib(), "trustyai", "dep")
        print(f"Installing Maven dependencies into {_ROOT}")
        os.system(f"mvn org.apache.maven.plugins:maven-dependency-plugin:2.10:get "
                  f"-DremoteRepositories=https://repository.sonatype.org/content/repositories/central  "
                  f"-Dartifact=org.kie.kogito:explainability-core:{TRUSTY_VERSION} -Dmaven.repo.local={_ROOT} -q")
        print(f"Installing Arrow dependencies into {_ROOT}")
        os.system(f"mvn org.apache.maven.plugins:maven-dependency-plugin:2.10:get "
                  f"-DremoteRepositories=https://repository.sonatype.org/content/repositories/central  "
                  f"-Dartifact=org.apache.arrow:arrow-vector:7.0.0 -Dmaven.repo.local={_ROOT} -q")

        _TESTS_FILE = os.path.join("org", "kie", "kogito", "explainability-core", TRUSTY_VERSION,
                                   f"explainability-core-{TRUSTY_VERSION}-tests.jar")
        os.system(f"wget -O {os.path.join(_ROOT, _TESTS_FILE)} https://repo1.maven.org/maven2/{_TESTS_FILE}")


setup(
    name="trustyai",
    version="0.2.5",
    description="Python bindings to the TrustyAI explainability library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trustyai-python/module",
    author="Rui Vieira",
    author_email="rui@redhat.com",
    license='Apache License 2.0',
    platforms='any',
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Java",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Java Libraries"
    ],
    package_data={
        "": ["*.jar"],
        "trustyai": ["dep/org/trustyai/arrow-converters-0.0.1.jar"],
    },
    packages=['trustyai', 'trustyai.model', 'trustyai.utils', 'trustyai.local', 'trustyai.metrics'],
    include_package_data=True,
    install_requires=['Jpype1', 'pyarrow'],
    cmdclass={"install": PostInstall},
)
