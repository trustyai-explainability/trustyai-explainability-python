Tutorial
===========
This page gives a brief tutorial on how to get started producing explanations with the TrustyAI
library.


Setup
-----
First, we'll need to import Numpy, which is what our example model and data will use.

.. code-block:: python
	:linenos:

	import numpy as np

For this example, we'll produce explanations of a simple, five feature linear model:

.. code-block:: python
	:linenos:
	:lineno-start: 2

	weights = np.random.uniform(low=-5, high=5, size=5)
	print(weights)

	def linear_model(x):
	    return np.dot(x, weights)

.. code-block:: console

	[ 0.48813504  2.15189366  1.02763376  0.44883183 -0.76345201]

Make note of the weights of our model; the second feature has the highest weight of ``2.15``, while
the fifth feature has the only negative weight of ``-.76``; we'll compare the explanations against
these weight values to see how the explanations reflect the model's behavior.

With our linear model defined, we need to wrap our model into a TrustyAI
:class:`~trustyai.model.Model` class. Here, we pass the linear model to the  ``predict_fun``
argument, indicating what function we are trying to explain:

.. code-block:: python
	:linenos:
	:lineno-start: 7

	from trustyai.model import Model
	model = Model(predict_fun=linear_model)

Finally, we'll establish the data point we are explaining by using the TrustyAI
:func:`~trustyai.model.simple_prediction` function. This wraps the input and output of our
model into a :class:`Prediction` object, which is what TrustyAI uses to produce explanations:

.. code-block:: python
	:linenos:
	:lineno-start: 9

	from trustyai.model import simple_prediction

	model_input = np.random.rand(1, 5)
	model_output = model(model_input)
	prediction = simple_prediction(model_input, model_output)

The above steps will look the same for nearly any desired use-case for any model and data.
In summary, we:

1) Wrapped the predict function into a :class:`~trustyai.model.Model`
2) Wrapped the desired prediction-to-be-explained into a :class:`Prediction`

LIME
----
With the setup complete, we can now produce explanations. We'll start with
LIME (`Local Interpretable Model-agnostic Explanations <https://arxiv.org/abs/1602.04938>`_), which
provide *saliencies*, weights associated with each input feature that describe how strongly
said feature contributed to the model's output. To do this, we'll first initialize the
:class:`~trustyai.explainers.LimeExplainer`. We'll set ``samples`` to 1000
(just picking an arbitrary large number, larger samples counts tend to produce better results
at the tradeoff of computation expense) and ``normalise_weights`` to False to return the raw
LIME saliencies:

.. code-block:: python
	:linenos:
	:lineno-start: 14

	from trustyai.explainers import LimeExplainer

	lime_explainer = LimeExplainer(samples=1000, normalise_weights=False)

Now we can produce and display the explanations:

.. code-block:: python
	:linenos:
	:lineno-start: 17

	lime_explanation = lime_explainer.explain(prediction, model)
	print(lime_explanation.as_dataframe())

.. code-block:: console
	:emphasize-lines: 3,6

	  output-0_features  output-0_saliency  output-0_value
	0           input-0        0.305466        0.645894
	1           input-1        0.902044        0.437587
	2           input-2        0.787208        0.891773
	3           input-3        0.370995        0.963663
	4           input-4       -0.280047        0.383442

Notice that the largest saliency is `input-1`: this makes sense, as it corresponds to the
largest weight in our linear model and thus had the greatest *positive* impact on the model output.
Meanwhile, `input-4` has the lowest saliency, and again this makes sense as it corresponds to the
only negative weight in our linear model, and thus this feature had the greatest *negative* impact
on the model output. This is the appeal of LIME; a quick and cheap way of producing *qualitative*
explanations of feature importance.

SHAP
----
Next, we'll produce some SHAP (`SHapley Additive exPlanations <https://arxiv.org/abs/1705.07874>`_)
explanations. SHAP provides *SHAP values*, which describe an additive explanation of the model
output; essentially a `receipt` for the model's output that shows how each feature's contribution
sums up to the final model output.

The process of generating a SHAP explanations looks very similar to LIME, with one main difference.
For SHAP, we need to define a *background dataset*, a set of representative datapoints to the model
that describe the model's *default* behavior. All explanations are then produced as comparisons
against this background dataset; i.e., how did the model perform differently for *this* datapoint
compared to the *background* dataset? In this case, we'll choose our background dataset to be all
zeros, as that provides the clearest baseline comparison against our desired explanation point.
We'll then pass the background when creating the :class:`~trustyai.explainers.SHAPExplainer`:

.. code-block:: python
	:linenos:
	:lineno-start: 19

	from trustyai.explainers import SHAPExplainer

	shap_explainer = SHAPExplainer(background=np.zeros([1, 5]))

Now we can produce and display the explanations:

.. code-block:: python
	:linenos:
	:lineno-start: 22

	explanation = explainer.explain(prediction, model)
	print(explanation.as_dataframe())

.. code-block:: console

	           Mean Background Value Feature Value  SHAP Value
	Background                     -             -    0.000000
	input-0                      0.0      0.645894    0.315284
	input-1                      0.0      0.437587    0.941641
	input-2                      0.0      0.891773    0.916416
	input-3                      0.0      0.963663    0.432523
	input-4                      0.0      0.383442   -0.292739
	Prediction                   0.0      2.313124    2.313124

Here, we notice the SHAP values exactly recover the product of each input feature
and the corresponding weight:

.. code-block:: console

	model_input * weights
	    [0.31528355  0.94164115  0.91641604  0.43252252 -0.2927392]

This makes sense; the exact contribution of each input to the output of the linear model
is the value of the feature multiplied by the corresponding weight. This is the advantage of
SHAP over LIME: rather than give qualitative measurements about a feature's contribution,
SHAP provides an estimate of the exact quantitative contribution, at the cost of being
much more computational expensive.


Counterfactuals
---------------
While SHAP and LIME produce explanations describing *how much* features contributed
to a model's output,Counterfactuals instead look to find ways of producing different outputs by
minimally modifying theinitial input. This is useful when looking for easy ways of achieving
specific desired results, answering "what is the smallest change I can make to get the result
I want?"

Producing counterfactual explanations is a little more involved than LIME or SHAP, because
we need to additionally specify feature *domains*, that is, the valid range of values which each
feature can possibly take on. This is to ensure all new feature values found by
the counterfactual explanation are "legal", and not things like negative age, "February 31", etc.

To do this, we wrap each of the inputs into :class:`Feature` objects via the
:func:`~trustyai.model.feature` function. For each :class:`Feature`, we'll need to provide a name,
a data type (in this case, ``"number"``), the original feature value,
and a ``domain`` that specifies the valid range of values. In this case, we'll constraint the
search to feature values between -10 and 10:

.. code-block:: python
	:linenos:
	:lineno-start: 24

	from trustyai.model import feature

	features = []
	for i, value in enumerate(model_input.reshape(-1)):
	    features.append(feature("Feature_{}".format(i), "number", value, domain=(-10., 10.))

Now, we use the :func:`~trustyai.model.counterfactual_prediction` function to wrap these features
with a counterfactual *goal*: the desired output we want to model to produce. Here, we'll select
``1.0`` as our goal, meaning the counterfactual explainer will try and find a set of inputs that
produce a model output of 1.0 ± 1%.

.. code-block:: python
	:linenos:
	:lineno-start: 29

	from trustyai.model import counterfactual_prediction

	prediction = counterfactual_prediction(
	    input_features=features,
	    outputs=np.array([[1.0]])
	)

We can now initialize the :class:`~trustyai.explainers.CounterfactualExplainer` and produce
explanations. We'll set ``steps`` to 10,000 in the explainer; this defines how many candidate
feature sets the counterfactual explainer will explore. In general, more steps produces better
results at the cost of compute time.

.. code-block:: python
	:linenos:
	:lineno-start: 35

	explainer = CounterfactualExplainer(steps=10_000)
	explanation = explainer.explain(prediction, model)
	print(explanation.as_dataframe())

.. code-block:: console
	:emphasize-lines: 2,4

	    features  proposed  original  constrained  difference
	0  feature_0 -2.023763  0.645894        False   -2.669657
	1  feature_1  0.437587  0.437587        False    0.000000
	2  feature_2  0.889143  0.891773        False   -0.002630
	3  feature_3  0.963663  0.963663        False    0.000000
	4  feature_4  0.383442  0.383442        False    0.000000

We can see that the counterfactual search has found an input that changes
``feature_0`` and ``feature_2``. We can then evaluate this counterfactual to see the new
model output:

.. code-block:: python
	:linenos:
	:lineno-start: 38

	print(model(explanation.proposed_features_array))

.. code-block:: console

	[1.0072685]

And indeed we've found a new input that produces an output of 1.0 ± 1%.
