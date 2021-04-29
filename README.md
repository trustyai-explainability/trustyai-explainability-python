[![Tests](https://github.com/ruivieira/python-trustyai/actions/workflows/workflow.yml/badge.svg)](https://github.com/ruivieira/python-trustyai/actions/workflows/workflow.yml)

# python-trustyai

Python bindings to [TrustyAI](https://kogito.kie.org/trustyai/)'s explainability library.

## Setup

The minimum dependencies can be installed with

```shell
pip install -r requirements.txt
```

If running the examples or developing, also install the development dependencies:

```shell
pip install -r requirements-dev.txt
```

## Getting started

To initialise, import the module and specify the localtion of the `explainability-core` JAR.
For instance,

```python
import trustyai

trustyai.init(path="/foo/bar/explainability-core-2.0.0-SNAPSHOT.jar")
```

## Examples

You can look at the [tests](./tests) for working examples.
