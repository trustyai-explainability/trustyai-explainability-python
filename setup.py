import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

TRUSTY_VERSION = "1.22.1.Final"

setup(
    name="trustyai",
    version="0.2.5",
    description="Python bindings to the TrustyAI explainability library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trustyai-python/module",
    author="Rui Vieira",
    author_email="rui@redhat.com",
    license="Apache License 2.0",
    platforms="any",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Java",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Java Libraries",
    ],
    package_data={
        "": ["*.jar"],
        "trustyai": ["dep/org/trustyai/*.jar"],
    },
    packages=[
        "trustyai",
        "trustyai.model",
        "trustyai.utils",
        "trustyai.local",
        "trustyai.metrics",
    ],
    include_package_data=True,
    install_requires=["Jpype1", "pyarrow"],
)
