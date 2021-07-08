[![Tests](https://github.com/ruivieira/python-trustyai/actions/workflows/workflow.yml/badge.svg)](https://github.com/ruivieira/python-trustyai/actions/workflows/workflow.yml) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ruivieira/python-trustyai/main)
# python-trustyai

Python bindings to [TrustyAI](https://kogito.kie.org/trustyai/)'s explainability library.

## Setup

### Local

The minimum dependencies can be installed with

```shell
pip install -r requirements.txt
```

If running the examples or developing, also install the development dependencies:

```shell
pip install -r requirements-dev.txt
```

### Docker

Alternatively create a container image and run it using

```shell
$ docker build -f Dockerfile -t ruivieira/python-trustyai:latest .
$ docker run --rm -it -p 8888:8888 ruivieira/python-trustyai:latest
```

The Jupyter server will be available at `localhost:8888`.

### Binder

You can also run the example Jupyter notebooks using `mybinder.org`:

- https://mybinder.org/v2/gh/ruivieira/python-trustyai/main

## Getting started

To initialise, import the module and specify the location of the `explainability-core` JAR.
For instance,

```python
import trustyai

trustyai.init(path="/foo/bar/explainability-core-2.0.0-SNAPSHOT.jar")
```

In order to get all the project's dependencies, the script `deps.sh` can be run and dependencies will
be stored locally under `./dep`.

This needs to be the very first call, before any other call to TrustyAI methods. After this, we can call all other methods, as shown in the examples.

### Writing your model in Python

To code a model in Python you need to write it a function with takes a Python list of `PredictionInput` and
returns a (Python) list of `PredictionOutput`. 

This function will then be passed as an argument to the Python `PredictionProvider`
which will take care of wrapping it in a Java `CompletableFuture` for you.
For instance,

```python
from trustyai.model import PredictionProvider

def myModelFunction(inputs):
    # do something with the inputs
    output = [predictionOutput1, predictionOutput2]
    return output

model = PredictionProvider(myModelFunction)

inputs = [predictionInput1, predictionInput2]

prediction = model.predictAsync(inputs).get()
```

You can see the `sumSkipModel` in the [LIME tests](./tests/test_limeexplainer.py).

## Examples

You can look at the [tests](./tests) for working examples.

There are also Jupyter notebooks available:

- [Counterfactual explanations](examples/Counterfactuals.ipynb)
