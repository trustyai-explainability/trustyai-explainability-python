# pylint: disable = consider-using-f-string
"""Info text used in Tyrus visualization explainer info"""
from trustyai.utils._visualisation import bold_red_html, bold_green_html

LIME_TEXT = """
<div style="padding: 10px">
    <h3> What is LIME?</h3>

    <p>
    LIME (<a href="https://arxiv.org/abs/1602.04938" target="_blank" rel="noopener noreferrer">Local Interpretable Model-agnostic Explanations</a>) explanations answer the following question:
    <blockquote>"Which features were most important to the predicted {{0}}?"</blockquote>
    LIME does this by providing <i>per-feature saliencies</i>, numeric weights that describe how strongly each feature contributed to the model’s output.
    </p>

    <p>
    In this plot, each horizontal bar represents a feature's saliency: features with positive importance to the predicted {{0}} are marked in {}, while
    features with negative importance are marked in {}. The larger the bar, the more important the feature was to the output. 
    </p>

    <p>
    To see how TrustyAI's LIME works, check out the <a href="https://trustyai-explainability-python.readthedocs.io/en/latest/generated/trustyai.explainers.LimeExplainer.html#trustyai.explainers.LimeExplainer"  target="_blank" rel="noopener noreferrer">documentation</a>!
    </p>
</div>
""".format(
    bold_green_html("green"), bold_red_html("red")
)

SHAP_TEXT = """
<div style="padding: 10px">
    <h3> What is SHAP?</h3>

    SHAP (<a href="https://arxiv.org/abs/1705.07874" target="_blank" rel="noopener noreferrer">SHapley Additive exPlanations</a>) explanations answer the following question:
    <blockquote>“By how much did each feature contribute to the predicted {{}}?”</blockquote>

    <p>
    SHAP does this by providing SHAP values that provide an additive explanation of the model output; a receipt for the model’s output. 
    SHAP will produce a list of <i>per-feature contributions</i>, the sum of which will equal the model's output. 
    To operate, SHAP also needs access to a <i>background dataset</i>, a set of representative input datapoints that captures 
    the model’s “normal behavior”. All SHAP values are comparisons against to this background data, i.e., 
    "By how much did each feature of this input contribute to the output, as compared to the background inputs?"
    </p>

    <p>
    In this plot, the dotted horizontal line shows the average model output over the background, the starting
    "baseline comparison" mark for a SHAP explanation. Then, each vertical bar or <i>candle</i> describes how 
    each feature {} or {} its contribution to the model's predicted output, marked by the solid horizontal line. The larger
    the feature's contribution, the larger the bar.
    </p>

    <p>
    To see how TrustyAI's SHAP works, check out the <a href="https://trustyai-explainability-python.readthedocs.io/en/latest/generated/trustyai.explainers.SHAPExplainer.html#trustyai.explainers.SHAPExplainer"  target="_blank" rel="noopener noreferrer">documentation</a>!
    </p>
</div>
""".format(
    bold_green_html("adds"), bold_red_html("subtracts")
)

CF_TEXT = """
<div style="padding: 10px">
    <h3> What is a Counterfactual?</h3>

    <p>
    Counterfactuals represent alternate, "what-if" scenarios; what other possible values of {0} 
    can be attained by modifying the input? 
    </p>

    <p>
    This plot shows all of counterfactuals
    produced during the computation of the LIME and SHAP explanations. On the x-axis are
    novel counterfactual values for {0}, while the y-axis shows the number of features that were changed to produce
    that particular value. Hover over individual points to see the exact changes to the original input
    necessary to produce the displayed counterfactual value of {0}. 
    </p>

    <p>
    To see how TrustyAI's Counterfactual Explainer works, check out the <a href="https://trustyai-explainability-python.readthedocs.io/en/latest/generated/trustyai.explainers.CounterfactualExplainer.html"  target="_blank" rel="noopener noreferrer">documentation</a>!
    </p>
</div>
"""
