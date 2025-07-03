![version](https://img.shields.io/badge/version-0.6.2-green) [![Tests](https://github.com/trustyai-python/module/actions/workflows/workflow.yml/badge.svg)](https://github.com/trustyai-python/examples/actions/workflows/workflow.yml)

# python-trustyai

Python bindings to [TrustyAI](https://kogito.kie.org/trustyai/)'s explainability library.

## Setup

### PyPi

Install from PyPi with

```shell
pip install trustyai
```

To install additional experimental features, also use

```shell
pip install trustyai[extras]
```

### Local

The minimum dependencies can be installed (from the root directory) with

```shell
pip install .
```

If running the examples or developing, also install the development dependencies:

```shell
pip install '.[dev]'
```

### Docker

Alternatively create a container image and run it using

```shell
$ docker build -f Dockerfile -t python-trustyai:latest .
$ docker run --rm -it -p 8888:8888 python-trustyai:latest
```

The Jupyter server will be available at `localhost:8888`.

### Binder

You can also run the example Jupyter notebooks
using `mybinder.org`: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/trustyai-python/trustyai-explainability-python-examples/main?labpath=examples)

## Documentation

Check out the [ReadTheDocs page](https://trustyai-explainability-python.readthedocs.io/en/latest/) for API references
and examples.

## Getting started

### Examples

There are several working examples available in the [examples](https://github.com/trustyai-explainability/trustyai-explainability-python-examples/tree/main/examples) repository.

## Contributing

Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for instructions on how to contribute to this project.