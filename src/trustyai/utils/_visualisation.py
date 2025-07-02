"""Visualiser utilies for explainer results"""

# pylint: disable = consider-using-f-string


# HTML FORMAT FUNCTIONS ============================================================================
def bold_green_html(content):
    """Format the content string as a bold, green html object"""
    return '<b style="color:{};">{}</b>'.format(
        DEFAULT_STYLE["positive_primary_colour"], content
    )


def bold_red_html(content):
    """Format the content string as a bold, red html object"""
    return '<b style="color:{};">{}</b>'.format(
        DEFAULT_STYLE["negative_primary_colour"], content
    )


def output_html(content):
    """Format the content string as a bold object in TrustyAI purple, used for
    Tyrus output displays"""
    return '<b style="color:#a64d79;background-color:#fff;">{}</b>'.format(content)


def feature_html(content):
    """Format the content string as a bold object in black, used for
    Tyrus feature displays"""
    return '<b style="color:#000000;background-color:#fff;">{}</b>'.format(content)


DEFAULT_STYLE = {
    "positive_primary_colour": "#13ba3c",
    "positive_primary_colour_faded": "#88dc9d",
    "negative_primary_colour": "#ee0000",
    "negative_primary_colour_faded": "#f67f7f",
    "neutral_primary_colour": "#ffffff",
}

DEFAULT_RC_PARAMS = {
    "patch.linewidth": 0.5,
    "patch.facecolor": "348ABD",
    "patch.edgecolor": "EEEEEE",
    "patch.antialiased": True,
    "font.size": 10.0,
    "axes.facecolor": "DDDDDD",
    "axes.edgecolor": "white",
    "axes.linewidth": 1,
    "axes.grid": True,
    "axes.titlesize": "x-large",
    "axes.labelsize": "large",
    "axes.labelcolor": "black",
    "axes.axisbelow": True,
    "text.color": "black",
    "xtick.color": "black",
    "xtick.direction": "out",
    "ytick.color": "black",
    "ytick.direction": "out",
    "legend.facecolor": "ffffff",
    "grid.color": "white",
    "grid.linestyle": "-",  # solid line
    "figure.figsize": (16, 9),
    "figure.dpi": 100,
    "figure.facecolor": "ffffff",
    "figure.edgecolor": "777777",
    "savefig.bbox": "tight",
}
